## ML Disease Prediction

### How to run
Create virtual env
```
python3 -m venv venv && source venv/bin/activate
```

Install dependencies
```
pip install -r requirements.txt
```

Migrate db
```
python manage.py migrate
```

Execute `disease-predict.ipynb`.

Run the server
```
python manage.py runserver
```
