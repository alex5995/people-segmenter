import os
import dropbox

if os.path.isfile('model.bin'):
    os.remove('model.bin')

token = os.environ.get('token')
dbx = dropbox.Dropbox(token, timeout=None)
dbx.files_download_to_file(download_path='./model.bin',
                           path='/people-segmenter-model.bin')