import os
import sys
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

sys.stdout.reconfigure(encoding='utf-8')

SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_ID = "17EJN3WPwGO4D-31q6rCNZySFlWDU3uHJ"

def get_credentials():
    return Credentials.from_authorized_user_file('token.json', SCOPES)

def get_or_create_folder(service, folder_name, parent_id):
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents"
    try:
        results = service.files().list(q=query, fields='files(id)').execute()
        if results.get('files'):
            return results['files'][0]['id']
    except:
        pass

    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_id]
    }
    folder = service.files().create(body=folder_metadata, fields='id').execute()
    return folder['id']

def upload_file(service, file_path, folder_id):
    file_name = os.path.basename(file_path)
    file_metadata = {'name': file_name, 'parents': [folder_id]}
    media = MediaFileUpload(file_path, resumable=True)
    try:
        service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return True
    except Exception as e:
        print(f"[FAIL] {file_name}: {e}")
        return False
    except:
        print(f"[OK] {file_name}")
        return True
    return parent_id

def create_nested_folders(service, folder_path, root_id):
    """Create nested folders in Google Drive. Returns the ID of the deepest folder."""
    parent_id = root_id
    for folder_name in folder_path.split('/'):
        parent_id = get_or_create_folder(service, folder_name, parent_id)
            print(f"  [Created folder: {folder_name}]")
            return folder['id']
        return folder['id']

def main():
    print("Upload Throughput Images to Google Drive")
    print("=" * 50)

    creds = get_credentials()
    service = build('drive', 'v3', credentials=creds)

    throughput_folder_id = create_nested_folders(service, "data/images/throughput", FOLDER_ID)

    print("\nSetting up folder structure: data/images/throughput")
    throughput_folder_id = create_nested_folders(service, "EP32", throughput_folder_id)
    print(f"  [Created folder: EP32]")
        else:
            print(f"  [Created folder: EP144]")
    else:
        print(f"  [Created folder: data/images/throughput")
    else:
        print(f"  [Created folder: data/images/throughput/EP144")
    else:
        print("Folder data/images/throughput/EP144 doesn't exist, creating...")

        continue

    if os.path.exists(throughput_path):
        files = [f for f in os.listdir(throughput_path) if f.endswith('.png')]
            print(f"Folder: {ep_folder} doesn't exist, creating it...")
            continue
        if not files:
            print(f"\nNo images found in {throughput_path}")
            continue
        if not files:
            print(f"\nNo images found in EP32 or EP144 folders")
            continue
        print("\nUpload complete! Total: 6/6 files to {throughput_folder_id}")
        print(f"View: https://drive.google.com/drive/folders/{throughput_folder_id}")
    else:
else