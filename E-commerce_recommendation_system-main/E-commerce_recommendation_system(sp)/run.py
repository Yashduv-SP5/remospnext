python -m venv myenv
myenv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install django
pip install numpy
pip install pandas
pip install joblib
pip install scikit-learn==1.2.2
python manage.py makemigrations
python manage.py migrate
python manage.py runserver

