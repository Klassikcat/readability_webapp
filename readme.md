# Readability App

<img src="readability_app/static/readability_icon.png">


![Python](https://img.shields.io/badge/python-3.x-blue.svg)
 [![All Contributors](https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square)](#contributors-)

"Readability" is a webapp measures readability both English and Korean using "XGBoost"

## Structure
```shell
readability_project
┖ readability_app
  ┖ models                    # Flask-SQLAlchemy Table 
    ┖ domain_table.py
    ┖ language_table.py
    ┖ text_table.py
  ┖ routes                    # Flask routes 
    ┖ metrics_route.py         
    ┖ main_route.py        
    ┖ refernce_route.py                
  ┖ static
    ┖ src                     # Javascript File 
      ┖ bootstrap.js 
      ┖ typewrite.js 
    ┖ form.css                # Input form css 
    ┖ style.css               # Global css 
    ┖ table.css               # Table css                 
  ┖ templates         
    ┖ about.html              
    ┖ base.html               # Base template     
    ┖ editor.html             # English/Korean predict Editor
    ┖ error.html              
    ┖ index.html              # Landing Page
    ┖ refernce.html           # Search, Delete 
    ┖ under_construction.html       
  ┖ utils       
    ┖ readability_pred.py     
  ┖ __init__.py               # webapp initializer       
  ┖ config.py                 # Configuration       

```
## Schema
<img src="./readability_app/static/schema.png">

## Stack

### front-end

<img alt="html5" src ="https://img.shields.io/badge/HTML5-E34F26.svg?&style=for-the-badge&logo=HTML5&logoColor=white"/> <img alt="css3" src ="https://img.shields.io/badge/CSS3-1572B6.svg?&style=for-the-badge&logo=CSS3&logoColor=white"/> <img alt="JavaScript" src ="https://img.shields.io/badge/JavaScript-F7DF1E.svg?&style=for-the-badge&logo=JavaScript&logoColor=black"/> <img alt="Flask" src ="https://img.shields.io/badge/Flask-000000.svg?&style=for-the-badge&logo=Flask&logoColor=white"/>

### back-end
<img alt="Python" src ="https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=Python&logoColor=white"/> <img alt="sqlite" src ="https://img.shields.io/badge/SQLite-003B57.svg?&style=for-the-badge&logo=SQLite&logoColor=white"/>


## ToDo
- [] Make Korean model 
- [] improve english model
- [] publish via server