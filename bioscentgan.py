

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import gradio as gr
from PIL import Image
from scipy import signal
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class SensoryFilters:
    """Animal-inspired sensory filter implementations"""

    @staticmethod
    def gabor_filter(image, frequency=0.1, theta=0, sigma=3, gamma=0.5):
        """Gabor filter for texture detection (dog-inspired micro-textures)"""
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size),
            sigma, theta,
            1.0/frequency, gamma,
            0, ktype=cv2.CV_32F
        )
        filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    @staticmethod
    def dog_microtexture_filter(image):
        """Dog-inspired: enhanced edge and texture detection using multiple Gabor filters"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        frequencies = [0.05, 0.1, 0.2]
        filtered_images = []

        for freq in frequencies:
            for theta in orientations:
                filtered = SensoryFilters.gabor_filter(gray, frequency=freq, theta=theta)
                filtered_images.append(filtered)

        combined = np.mean(filtered_images, axis=0)

        if len(image.shape) == 3:
            combined = cv2.cvtColor(combined.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        return combined.astype(np.uint8)

    @staticmethod
    def eagle_vision_filter(image):
        """Eagle-inspired: ultra-high contrast and sharpness enhancement"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
        else:
            l = image

        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l)

        kernel_sharpen = np.array([[-1,-1,-1],
                                   [-1, 9,-1],
                                   [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced_l, -1, kernel_sharpen)

        if len(image.shape) == 3:
            enhanced_lab = cv2.merge([sharpened, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            enhanced = sharpened

        return enhanced

    @staticmethod
    def bee_uv_filter(image):
        """Bee-inspired: UV spectrum simulation using blue channel enhancement"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        blue_channel = image[:, :, 0]
        enhanced_blue = cv2.equalizeHist(blue_channel)

        uv_sim = cv2.addWeighted(enhanced_blue, 0.6, v, 0.4, 0)

        enhanced_hsv = cv2.merge([h, s, uv_sim])
        enhanced = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

        return enhanced

    @staticmethod
    def steerable_filter(image, angle=0):
        """Steerable filter for directional feature enhancement"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        ksize = 5
        sigma = 1.0

        x, y = np.meshgrid(np.arange(-ksize//2+1, ksize//2+1),
                          np.arange(-ksize//2+1, ksize//2+1))

        theta = angle
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)

        gaussian = np.exp(-(x_theta**2 + y_theta**2) / (2 * sigma**2))
        derivative = -x_theta / sigma**2 * gaussian

        kernel = derivative / np.sum(np.abs(derivative))
        filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)

        normalized = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)

        if len(image.shape) == 3:
            normalized = cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        return normalized.astype(np.uint8)


class Generator(nn.Module):
    """Simplified Generator for face enhancement"""

    def __init__(self, input_channels=3, output_channels=3):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, output_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class FaceFeatureExtractor(nn.Module):
    """Simplified feature extractor inspired by ArcFace/FaceNet"""

    def __init__(self, embedding_size=512):
        super(FaceFeatureExtractor, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.embedding = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        embedding = self.embedding(features)
        return F.normalize(embedding, p=2, dim=1)


class BioScentGAN:
    """Main BioScentGAN system"""

    def __init__(self, device='cpu'):
        self.device = device
        self.generator = Generator().to(device)
        self.feature_extractor = FaceFeatureExtractor().to(device)
        self.database = []
        self.database_features = []
        self.database_metadata = []

        self.generator.eval()
        self.feature_extractor.eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def preprocess_image(self, image):
        """Preprocess input image"""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

        return self.transform(image).unsqueeze(0).to(self.device)

    def apply_sensory_filters(self, image):
        """Apply all animal-inspired sensory filters"""
        if isinstance(image, Image.Image):
            image = np.array(image)

        dog_filtered = SensoryFilters.dog_microtexture_filter(image)
        eagle_filtered = SensoryFilters.eagle_vision_filter(image)
        bee_filtered = SensoryFilters.bee_uv_filter(image)
        steerable_filtered = SensoryFilters.steerable_filter(image, angle=np.pi/4)

        combined = cv2.addWeighted(dog_filtered, 0.3, eagle_filtered, 0.3, 0)
        combined = cv2.addWeighted(combined, 1.0, bee_filtered, 0.2, 0)
        combined = cv2.addWeighted(combined, 1.0, steerable_filtered, 0.2, 0)

        return {
            'dog': dog_filtered,
            'eagle': eagle_filtered,
            'bee': bee_filtered,
            'steerable': steerable_filtered,
            'combined': combined
        }

    def enhance_face(self, image):
        """Enhance degraded face using GAN"""
        preprocessed = self.preprocess_image(image)

        with torch.no_grad():
            enhanced = self.generator(preprocessed)

        enhanced_np = enhanced.squeeze(0).cpu().numpy()
        enhanced_np = (enhanced_np * 0.5 + 0.5) * 255
        enhanced_np = np.transpose(enhanced_np, (1, 2, 0))
        enhanced_np = np.clip(enhanced_np, 0, 255).astype(np.uint8)

        return enhanced_np

    def extract_features(self, image):
        """Extract feature embedding from image"""
        preprocessed = self.preprocess_image(image)

        with torch.no_grad():
            features = self.feature_extractor(preprocessed)

        return features.cpu().numpy()

    def add_to_database(self, image, metadata=None):
        """Add a face to the database"""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            image = Image.fromarray(image)

        features = self.extract_features(image)

        self.database.append(image)
        self.database_features.append(features)
        self.database_metadata.append(metadata or {})

        return len(self.database) - 1

    def match_face(self, query_image, top_k=3):
        """Match query face to database using cosine similarity"""
        if len(self.database) == 0:
            return []

        query_features = self.extract_features(query_image)
        database_features = np.vstack(self.database_features)

        similarities = cosine_similarity(query_features, database_features)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        matches = []
        for idx in top_indices:
            matches.append({
                'index': int(idx),
                'similarity': float(similarities[idx]),
                'image': self.database[idx],
                'metadata': self.database_metadata[idx]
            })

        return matches

    def generate_visualization(self, original_image, save_path=None):
        """Generate comprehensive visualization"""
        if isinstance(original_image, np.ndarray):
            if len(original_image.shape) == 2:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
            original_pil = Image.fromarray(original_image)
        else:
            original_pil = original_image
            original_image = np.array(original_image)

        sensory_filtered = self.apply_sensory_filters(original_image)
        enhanced = self.enhance_face(sensory_filtered['combined'])
        matches = self.match_face(enhanced, top_k=3)

        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 5, figure=fig, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(original_image)
        ax1.set_title('Original (Degraded)', fontsize=12, fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(sensory_filtered['dog'])
        ax2.set_title('Dog Filter\n(Micro-textures)', fontsize=10)
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(sensory_filtered['eagle'])
        ax3.set_title('Eagle Filter\n(High Contrast)', fontsize=10)
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[0, 3])
        ax4.imshow(sensory_filtered['bee'])
        ax4.set_title('Bee Filter\n(UV Simulation)', fontsize=10)
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[0, 4])
        ax5.imshow(sensory_filtered['steerable'])
        ax5.set_title('Steerable Filter\n(Directional)', fontsize=10)
        ax5.axis('off')

        ax6 = fig.add_subplot(gs[1, 0])
        ax6.imshow(sensory_filtered['combined'])
        ax6.set_title('Combined Sensory\nFilters', fontsize=12, fontweight='bold')
        ax6.axis('off')

        ax7 = fig.add_subplot(gs[1, 1])
        ax7.imshow(enhanced)
        ax7.set_title('GAN Enhanced', fontsize=12, fontweight='bold')
        ax7.axis('off')

        if matches:
            for i, match in enumerate(matches):
                ax = fig.add_subplot(gs[1, 2+i])
                match_img = np.array(match['image'])
                ax.imshow(match_img)
                metadata_str = f"Match {i+1}\nSimilarity: {match['similarity']:.3f}"
                if match['metadata']:
                    for key, value in match['metadata'].items():
                        metadata_str += f"\n{key}: {value}"
                ax.set_title(metadata_str, fontsize=9)
                ax.axis('off')

        query_features = self.extract_features(enhanced)
        ax8 = fig.add_subplot(gs[2, :2])
        feature_map = query_features.reshape(-1, 32)
        im = ax8.imshow(feature_map, cmap='viridis', aspect='auto')
        ax8.set_title('Feature Embedding Heatmap', fontsize=12, fontweight='bold')
        ax8.set_xlabel('Feature Dimension')
        ax8.set_ylabel('Feature Group')
        plt.colorbar(im, ax=ax8)

        if len(matches) > 0:
            ax9 = fig.add_subplot(gs[2, 2:])
            match_names = [f"Match {i+1}" for i in range(len(matches))]
            similarities = [m['similarity'] for m in matches]
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(matches)))
            bars = ax9.barh(match_names, similarities, color=colors)
            ax9.set_xlabel('Cosine Similarity', fontsize=11)
            ax9.set_title('Top-3 Match Similarities', fontsize=12, fontweight='bold')
            ax9.set_xlim([0, 1])
            for i, (bar, sim) in enumerate(zip(bars, similarities)):
                ax9.text(sim + 0.02, i, f'{sim:.3f}', va='center', fontsize=10)

        plt.suptitle('BioScentGAN: Animal-Inspired Face Enhancement & Matching',
                     fontsize=16, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, enhanced, matches


def create_sample_database(gan_system, num_samples=10):
    """Create a sample database with synthetic faces"""
    print("Creating sample database...")

    for i in range(num_samples):
        face = np.random.randint(100, 200, (128, 128, 3), dtype=np.uint8)

        cv2.circle(face, (64, 50), 20, (200, 180, 160), -1)
        cv2.ellipse(face, (55, 48), (3, 5), 0, 0, 360, (50, 50, 50), -1)
        cv2.ellipse(face, (73, 48), (3, 5), 0, 0, 360, (50, 50, 50), -1)
        cv2.ellipse(face, (64, 70), (8, 5), 0, 0, 180, (100, 50, 50), 2)

        metadata = {
            'id': f'person_{i:03d}',
            'hair': np.random.choice(['black', 'brown', 'blonde', 'red']),
            'skin': np.random.choice(['light', 'medium', 'dark']),
            'eyes': np.random.choice(['brown', 'blue', 'green', 'hazel'])
        }

        gan_system.add_to_database(face, metadata)

    print(f"Database created with {num_samples} entries")


def gradio_interface():
    """Create Gradio interface for BioScentGAN"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gan_system = BioScentGAN(device=device)

    create_sample_database(gan_system, num_samples=15)

    def process_image(image, hair_color, skin_tone, eye_color):
        """Process image through BioScentGAN pipeline"""
        if image is None:
            return None, "Please upload an image"

        metadata = {}
        if hair_color:
            metadata['hair'] = hair_color
        if skin_tone:
            metadata['skin'] = skin_tone
        if eye_color:
            metadata['eyes'] = eye_color

        fig, enhanced, matches = gan_system.generate_visualization(image)

        result_text = "### Processing Results\n\n"
        result_text += f"**Enhanced image generated using animal-inspired filters**\n\n"

        if matches:
            result_text += "#### Top Matches:\n"
            for i, match in enumerate(matches, 1):
                result_text += f"\n**Match {i}** (Similarity: {match['similarity']:.3f})\n"
                if match['metadata']:
                    for key, value in match['metadata'].items():
                        result_text += f"  - {key}: {value}\n"
        else:
            result_text += "\nNo matches found in database."

        return fig, result_text

    with gr.Blocks(title="BioScentGAN", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧬 BioScentGAN: Animal-Inspired Face Enhancement

        Upload a degraded face image to enhance it using bio-inspired sensory filters and match it to a database.

        **Features:**
        - 🐕 Dog micro-texture detection (Gabor filters)
        - 🦅 Eagle ultra-high contrast enhancement
        - 🐝 Bee UV spectrum simulation
        - 🔄 Steerable directional filters
        - 🤖 GAN-based reconstruction
        - 🎯 Cosine similarity matching
        """)

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(label="Upload Degraded Face", type="numpy")

                gr.Markdown("### Optional Metadata")
                hair_input = gr.Dropdown(
                    choices=['', 'black', 'brown', 'blonde', 'red'],
                    label="Hair Color"
                )
                skin_input = gr.Dropdown(
                    choices=['', 'light', 'medium', 'dark'],
                    label="Skin Tone"
                )
                eye_input = gr.Dropdown(
                    choices=['', 'brown', 'blue', 'green', 'hazel'],
                    label="Eye Color"
                )

                process_btn = gr.Button("🚀 Process Image", variant="primary")

            with gr.Column(scale=2):
                output_plot = gr.Plot(label="Visualization")
                output_text = gr.Markdown()

        process_btn.click(
            fn=process_image,
            inputs=[image_input, hair_input, skin_input, eye_input],
            outputs=[output_plot, output_text]
        )

        gr.Markdown("""
        ---
        ### About the Filters
        - **Dog Filter**: Uses Gabor filters at multiple orientations and frequencies to detect fine textures
        - **Eagle Filter**: Applies CLAHE and sharpening for ultra-high contrast enhancement
        - **Bee Filter**: Simulates UV spectrum sensitivity using blue channel enhancement
        - **Steerable Filter**: Directional derivative filters for edge enhancement
        """)

    return demo


def main():
    """Main entry point"""
    print("=" * 60)
    print("BioScentGAN: Animal-Inspired Face Enhancement System")
    print("=" * 60)
    print(f"\nDevice: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("\nStarting Gradio interface...")

    demo = gradio_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()