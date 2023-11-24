import os
import src.thisdir
homedir =  os.path.expanduser(r"~")
thisdir = src.thisdir.thisdir()
def check_for_write_permissions(dir):
        if 'FLATPAK_ID' in os.environ:
            import subprocess

            command = f'cat /var/lib/flatpak/app/io.github.tntwise.REAL-Video-Enhancer/x86_64/master/active/metadata'

            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = result.stdout.split('\n')
            output_2=[]
            for i in output:
                if len(i) > 0 and i != '\n':
                    output_2.append(i)
            directories_with_permissions=[]
            for i in output_2:
                print(i)
                if 'filesystems=' in i:
                    i=i.split(';')
                    print(i)
                    s=[]
                    for e in  i:
                        if len(e) > 0 and i != '\n':
                            s.append(e)
                    for j in s:
                        j=j.replace('filesystems=','')
                        j=j.replace('xdg-',f'{homedir}/')
                        
                        directories_with_permissions.append(j)
                    break
            for i in directories_with_permissions:
                if i.lower() in dir.lower() or 'io.github.tntwise.real-video-enhancer' in dir.lower():
                    print(f'I: {i}')
                    print(f'Dir: {dir}')
                    return True
                else:
                    print(f'Dir: {dir}')
                    print(f'I: {i}')
                    i=i.replace(f'{homedir}','/run/user/1000/doc/fecc5049/')
                    if i.lower() in dir.lower() or 'io.github.tntwise.real-video-enhancer' in dir.lower():
                        print(f'I: {i}')
                        print(f'Dir: {dir}')
                        return True
                    return False
        else:
                if os.access(dir, os.R_OK) and os.access(dir, os.W_OK):
                    print('has access')
                    return True
                return False