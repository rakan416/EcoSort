path = 'Data Python\Training_Code\Training Django\django_venv\Scripts\python.exe'
ttk = path.index('.')
pathpred = path[:ttk]+'-predict'+path[ttk:]
print(pathpred)