import os
import dropbox

token = os.environ.get('token')
dbx = dropbox.Dropbox(token, timeout=None)
dbx.files_download_to_file(download_path='./model.bin',
                           path='/people-segmenter-model.bin')