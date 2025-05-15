import tensorflow as tf
from tensorflow.keras import layers

class DataAugmentation:
    """
    Class to handle data augmentation for image datasets
    
    Augmentation is applied at the individual image level during dataset mapping,
    which happens when the dataset is consumed (during training).
    Each time an image is fetched from the dataset, a new random augmentation is applied.
    """
    
    def __init__(self, rotation_factor=0.50, zoom_factor=0.10, translation_factor=0.10):
        """
        Initialize data augmentation with configurable parameters
        
        Args:
            rotation_factor: Maximum rotation angle in radians
            zoom_factor: Maximum zoom factor
            translation_factor: Maximum translation as fraction of image dimensions
        """
        self.rotation_factor = rotation_factor
        self.zoom_factor = zoom_factor
        self.translation_factor = translation_factor
        self.augmentation_pipeline = self._create_pipeline()
        
    def _create_pipeline(self):
        """Create the augmentation pipeline with current parameters"""
        return tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(self.rotation_factor),
            # layers.RandomZoom(self.zoom_factor),
            # layers.RandomTranslation(self.translation_factor, self.translation_factor),
        ])
    
    def augment(self, images, training=True):
        """
        Apply augmentation to images
        
        This is called for each individual image during dataset mapping.
        A new random augmentation is applied each time this is called.
        """
        return self.augmentation_pipeline(images, training=training)
    
    
class AggressiveDataAugmentation:
    """
    Class to handle more aggressive data augmentation with RandAugment-style operations
    
    Like the standard DataAugmentation class, augmentation is applied at the individual image level
    during dataset mapping. Each time an image is fetched from the dataset, a new random
    augmentation is applied with more aggressive transformations.
    """
    
    def __init__(self, rotation_factor=0.10, zoom_factor=0.10, translation_factor=0.10, image_size=32, crop_padding=8, num_ops=2):
        """
        Initialize aggressive data augmentation with configurable parameters
        
        Args:
            rotation_factor: Maximum rotation angle in radians (for compatibility with DataAugmentation)
            zoom_factor: Maximum zoom factor (for compatibility with DataAugmentation)
            translation_factor: Maximum translation as fraction of dimensions (for compatibility with DataAugmentation)
            image_size: Size of input images (assumes square)
            crop_padding: Amount of padding to add before random crop
            num_ops: Number of random augmentation ops to apply
        """
        # Store parameters for compatibility with DataAugmentation
        self.rotation_factor = rotation_factor
        self.zoom_factor = zoom_factor
        self.translation_factor = translation_factor
        
        # Additional parameters for aggressive augmentation
        self.image_size = image_size
        self.crop_padding = crop_padding
        self.num_ops = num_ops
        
        # Create layers
        self.affine_layer = self._create_affine_layer()
        self.augmentation_pipeline = self._create_pipeline()
        
    def _create_affine_layer(self):
        """Create the affine transformation layer"""
        return tf.keras.Sequential([
            layers.RandomRotation(0.1),          # ~±18°
            layers.RandomZoom(0.1, 0.1),          # ±10% zoom
            layers.RandomTranslation(0.1, 0.1),   # ±10% shift
        ])
    
    def _create_pipeline(self):
        """Create a Sequential model for compatibility with the original class"""
        # This is a dummy pipeline that will not be used directly
        # but allows the class to have the same interface
        return self.affine_layer
    
    def augment(self, images, training=True):
        """
        Apply augmentation to images
        
        This is called for each individual image during dataset mapping.
        A new random augmentation is applied each time this is called.
        
        Args:
            images: Batch of images to augment
            training: Whether to apply augmentation (only applied during training)
            
        Returns:
            Augmented images
        """
        if not training:
            return images
            
        # For single image input
        if len(tf.shape(images)) == 3:
            return self._augment_single(images)
        # For batch of images
        else:
            return tf.map_fn(self._augment_single, images)
    
    @tf.function
    def _augment_single(self, image):
        """
        Apply aggressive augmentation to a single image
        
        This applies a sequence of transformations to each individual image:
        1. Random crop with padding and horizontal flip
        2. Affine transformations (rotation, zoom, translation)
        3. RandAugment-style operations (randomly selected)
        """
        # Step 1: Pad + crop + flip
        padded_size = self.image_size + self.crop_padding
        image = tf.image.resize_with_crop_or_pad(image, padded_size, padded_size)
        image = tf.image.random_crop(image, size=[self.image_size, self.image_size, 3])
        image = tf.image.random_flip_left_right(image)

        # Step 2: Apply keras.layers affine transforms (GPU-efficient)
        image = self.affine_layer(tf.expand_dims(image, 0), training=True)[0]

        # Step 3: Apply RandAugment-style custom ops
        image = self._rand_augment(image, N=self.num_ops)

        return image
    
    @tf.function
    def _rand_augment(self, image, N=2):
        """
        Apply N randomly selected augmentation operations
        
        This randomly selects N operations from the available transformations
        and applies them in sequence to the image.
        """
        def rotate(img):
            flip = tf.random.uniform((), 0, 1)
            return tf.image.rot90(img, k=tf.cast(flip > 0.5, tf.int32))

        def translate(img):
            max_shift = 4  # more aggressive shift
            dx = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
            dy = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
            img = tf.image.pad_to_bounding_box(img, max_shift, max_shift, 
                                              self.image_size + 2 * max_shift, 
                                              self.image_size + 2 * max_shift)
            img = tf.image.crop_to_bounding_box(img, max_shift + dy, max_shift + dx, 
                                               self.image_size, self.image_size)
            return img

        def contrast(img):
            return tf.image.random_contrast(img, 0.6, 1.4)  

        def brightness(img):
            return tf.image.random_brightness(img, 0.2) 

        def saturation(img):
            return tf.image.random_saturation(img, 0.6, 1.6)  

        def hue(img):
            return tf.image.random_hue(img, 0.08)  

        all_ops = [rotate, translate, contrast, brightness, saturation, hue]
        indices = tf.random.shuffle(tf.range(len(all_ops)))[:N]

        for i in indices:
            image = tf.switch_case(i, branch_fns=[lambda i=i: all_ops[i](image) for i in range(len(all_ops))])

        image = tf.clip_by_value(image, 0.0, 1.0)
        return image
    
    