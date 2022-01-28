from setuptools import setup

setup(
    name='pytima',
    version='0.0.1',    
    description='Reads tescan tima mindif data structures',
    url='https://github.com/BenFGA/pytima',
    author='Ben Chi',
    author_email='ben@fractalgeoanalytics.com',
    license='MIT',
    packages=['pytima'],
    install_requires=['wheel','numpy','scikit-image','matplotlib','pandas','imagecodecs'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.10',
    ])
