from google.cloud import storage
import os

def upload_local_directory_to_gcs(local_path, bucket_name, gcs_path):
    """로컬 디렉토리와 모든 하위 디렉토리를 GCS 버킷에 업로드하는 함수"""
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for root, dirs, files in os.walk(local_path):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            
            # GCS의 목적지 경로 설정
            relative_path = os.path.relpath(local_file_path, local_path)
            remote_path = os.path.join(gcs_path, relative_path)

            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file_path)

            print(f"File {local_file_path} uploaded to gs://{bucket_name}/{remote_path}")

# 사용 예시
# local_directory = "/path/to/local/directory"
# bucket_name = "your-bucket-name"
# gcs_path = "path/in/bucket"

# upload_local_directory_to_gcs(local_directory, bucket_name, gcs_path)