from setuptools import find_packages,setup

def get_req(file_path:str)->list[str]: # return list of req in req.txt
    requirements=[]
    with open(file_path) as file:
        requirements=file.readlines()
        requirements=[req.replace('\n',"") for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
        return requirements
 


setup(
    name='my-project',
    version='0.0.1',
    author='Krish Kataria',
    author_email='kkrishkataria@gmail.com',
    packages=find_packages(),
    install_requires=get_req('requirements.txt')
)