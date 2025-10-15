import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import hog, local_binary_pattern
from collections import defaultdict
import json
from datetime import datetime


class FaceEnhancer:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def enhance_image(self, image):
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            enhanced_lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            enhanced = self.clahe.apply(image)

        enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)

        return sharpened


class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    def detect_and_align(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            return None, None

        x, y, w, h = faces[0]
        face_roi = image[y:y+h, x:x+w]

        face_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(face_gray)

        if len(eyes) >= 2:
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            left_eye = eyes_sorted[0]
            right_eye = eyes_sorted[1]

            left_eye_center = (left_eye[0] + left_eye[2] // 2, left_eye[1] + left_eye[3] // 2)
            right_eye_center = (right_eye[0] + right_eye[2] // 2, right_eye[1] + right_eye[3] // 2)

            dx = right_eye_center[0] - left_eye_center[0]
            dy = right_eye_center[1] - left_eye_center[1]
            angle = np.degrees(np.arctan2(dy, dx))

            center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                     (left_eye_center[1] + right_eye_center[1]) // 2)

            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aligned = cv2.warpAffine(face_roi, M, (w, h))

            return aligned, (x, y, w, h)

        return face_roi, (x, y, w, h)


class FeatureExtractor:
    def __init__(self):
        self.target_size = (128, 128)

    def extract_color_histogram(self, image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [60], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [60], [0, 256])

        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()

        return np.concatenate([hist_h, hist_s, hist_v])

    def extract_hog_features(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        resized = cv2.resize(gray, self.target_size)

        features = hog(
            resized,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False
        )

        return features

    def extract_lbp_features(self, image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        resized = cv2.resize(gray, self.target_size)

        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(resized, n_points, radius, method='uniform')

        n_bins = n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

        return hist

    def extract_all_features(self, face_image):
        color_hist = self.extract_color_histogram(face_image)
        hog_features = self.extract_hog_features(face_image)
        lbp_features = self.extract_lbp_features(face_image)

        combined_features = np.concatenate([color_hist, hog_features, lbp_features])

        return combined_features


class FacialRecognitionSystem:
    def __init__(self, database_path='face_database'):
        self.database_path = Path(database_path)
        self.database_path.mkdir(exist_ok=True)

        self.detector = FaceDetector()
        self.enhancer = FaceEnhancer()
        self.feature_extractor = FeatureExtractor()

        self.database = {
            'features': [],
            'metadata': []
        }

        self.load_database()

    def load_database(self):
        db_file = self.database_path / 'database.pkl'
        if db_file.exists():
            with open(db_file, 'rb') as f:
                self.database = pickle.load(f)
            print(f"Loaded database with {len(self.database['features'])} entries")
        else:
            print("No existing database found. Starting fresh.")

    def save_database(self):
        db_file = self.database_path / 'database.pkl'
        with open(db_file, 'wb') as f:
            pickle.dump(self.database, f)
        print(f"Database saved with {len(self.database['features'])} entries")

    def register_person(self, image_path, name, person_type='missing', additional_info=None):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return False

        face, bbox = self.detector.detect_and_align(image)
        if face is None:
            print(f"Error: No face detected in {image_path}")
            return False

        enhanced_face = self.enhancer.enhance_image(face)

        features = self.feature_extractor.extract_all_features(enhanced_face)

        metadata = {
            'name': name,
            'type': person_type,
            'registered_date': datetime.now().isoformat(),
            'image_path': str(image_path),
            'additional_info': additional_info or {}
        }

        self.database['features'].append(features)
        self.database['metadata'].append(metadata)

        self.save_database()
        print(f"Successfully registered: {name} ({person_type})")
        return True

    def search_face(self, query_image_path, top_k=5, threshold=0.65):
        query_image = cv2.imread(str(query_image_path))
        if query_image is None:
            print(f"Error: Could not load query image {query_image_path}")
            return []

        face, bbox = self.detector.detect_and_align(query_image)
        if face is None:
            print("Error: No face detected in query image")
            return []

        enhanced_face = self.enhancer.enhance_image(face)

        query_features = self.feature_extractor.extract_all_features(enhanced_face)

        if len(self.database['features']) == 0:
            print("Database is empty. No matches to compare.")
            return []

        db_features = np.array(self.database['features'])
        query_features = query_features.reshape(1, -1)

        similarities = cosine_similarity(query_features, db_features)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            if similarity >= threshold:
                result = {
                    'confidence': float(similarity * 100),
                    'metadata': self.database['metadata'][idx],
                    'similarity_score': float(similarity)
                }
                results.append(result)

        return results

    def display_results(self, query_image_path, results):
        query_image = cv2.imread(str(query_image_path))

        if len(results) == 0:
            print("\nNo matches found above threshold.")
            cv2.imshow('Query Image', query_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return

        print(f"\n{'='*70}")
        print(f"SEARCH RESULTS - Found {len(results)} match(es)")
        print(f"{'='*70}\n")

        for i, result in enumerate(results, 1):
            metadata = result['metadata']
            confidence = result['confidence']

            print(f"Match #{i}:")
            print(f"  Name: {metadata['name']}")
            print(f"  Type: {metadata['type']}")
            print(f"  Confidence: {confidence:.2f}%")
            print(f"  Registered: {metadata['registered_date']}")

            if metadata['additional_info']:
                print(f"  Additional Info:")
                for key, value in metadata['additional_info'].items():
                    print(f"    - {key}: {value}")
            print()

        cv2.imshow('Query Image', query_image)

        for i, result in enumerate(results[:3], 1):
            match_image = cv2.imread(result['metadata']['image_path'])
            if match_image is not None:
                confidence = result['confidence']
                cv2.putText(
                    match_image,
                    f"{confidence:.1f}% Match",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                cv2.imshow(f'Match #{i} - {result["metadata"]["name"]}', match_image)

        print(f"{'='*70}")
        print("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    system = FacialRecognitionSystem()

    print("\n" + "="*70)
    print("FACIAL RECOGNITION SYSTEM FOR MISSING PERSONS")
    print("="*70 + "\n")

    while True:
        print("\nOptions:")
        print("1. Register Missing Person")
        print("2. Register Unidentified Body")
        print("3. Search for Match")
        print("4. View Database Stats")
        print("5. Exit")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == '1':
            image_path = input("Enter image path: ").strip()
            name = input("Enter person's name: ").strip()
            age = input("Enter age (optional): ").strip()
            location = input("Enter last seen location (optional): ").strip()

            additional_info = {}
            if age:
                additional_info['age'] = age
            if location:
                additional_info['last_seen'] = location

            system.register_person(image_path, name, 'missing', additional_info)

        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            case_id = input("Enter case ID: ").strip()
            location = input("Enter location found (optional): ").strip()
            date_found = input("Enter date found (optional): ").strip()

            additional_info = {'case_id': case_id}
            if location:
                additional_info['location_found'] = location
            if date_found:
                additional_info['date_found'] = date_found

            system.register_person(image_path, f"Unidentified-{case_id}", 'unidentified', additional_info)

        elif choice == '3':
            query_path = input("Enter query image path: ").strip()
            top_k = input("Number of top matches to show (default 5): ").strip()
            top_k = int(top_k) if top_k.isdigit() else 5

            threshold = input("Minimum confidence threshold 0-1 (default 0.65): ").strip()
            try:
                threshold = float(threshold) if threshold else 0.65
            except ValueError:
                threshold = 0.65

            results = system.search_face(query_path, top_k, threshold)
            system.display_results(query_path, results)

        elif choice == '4':
            total = len(system.database['features'])
            missing = sum(1 for m in system.database['metadata'] if m['type'] == 'missing')
            unidentified = sum(1 for m in system.database['metadata'] if m['type'] == 'unidentified')

            print(f"\nDatabase Statistics:")
            print(f"  Total entries: {total}")
            print(f"  Missing persons: {missing}")
            print(f"  Unidentified bodies: {unidentified}")

        elif choice == '5':
            print("\nExiting system. Goodbye!")
            break

        else:
            print("\nInvalid choice. Please try again.")


if __name__ == "__main__":
    main()
