# # Previously run this in console:
# python -m venv /home/user/.venv/ && source /home/user/.venv/bin/activate && pip install -r requirements.txt

# export PYTHONDONTWRITEBYTECODE=1 
# export PATH=$PATH:/home/user/.local/bin

if __name__ == '__main__':

    
    from subprocess import run
    run("gunicorn -b 0.0.0.0:8080 app --reload".split(' '))

