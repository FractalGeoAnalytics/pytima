from setuptools import setup

setup(
    name='pytima',
    version='0.0.1',    
    description='A example Python package',
    url='https://github.com/shuds13/pyexample',
    author='Ben Chi',
    author_email='ben@fractalgeoanalytics.com',
    license='MIT',
    packages=['pytima'],
    install_requires=['numpy'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.10',
    ])
