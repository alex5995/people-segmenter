import os
import dropbox

model_dir = 'people-segmenter-model'
remote_dir = '/people-segmenter-model'

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)

dbx = dropbox.Dropbox('sl.AloiwVtAIn2Fllj0YaetjAw5RV8EnL_2RyVaWvWQXDV5OJWaiD62zCkECJrTBZF9q2CN-Yf-57fIO1lN08AGMTU4bFHhjXhP6pHZTHfRh6I3hSmT7QJdHIAh-wys_bMvVPZ9F-d7o5w', timeout=None)
dbx.files_download_to_file(download_path=os.path.join(model_dir, 'model.bin'),
                           path=os.path.join(remote_dir, 'model.bin'))