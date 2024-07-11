import setuptools

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name='coursekg',
    version='0.0.1',
    author='wangtao',
    author_email='wangtao.cpu@gmail.com',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/wangtao2001/CourseKG',
    packages=setuptools.find_packages(),
    install_requires=[
        'selenium>=4.15.2',
        'requests>=2.31.0'
    ]
)