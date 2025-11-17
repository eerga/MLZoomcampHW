# 🏠 Machine Learning Project Predicting Real Estate Prices for Residents in Boston

[Midterm project](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/projects) for DataTalks.Club Machine Learning ZoomCamp 2025. 

## ⚠️ Problem Statement
The Boston real estate market has become increasingly challenging to navigate, with property prices reaching unprecedented levels and making homeownership a distant dream for many. Whether you're a first-time homebuyer trying to find an affordable entry point or an investor seeking rental income opportunities, understanding what drives property values is crucial for making informed decisions in this competitive market.

The biggest challenge facing potential buyers in Boston is the difficulty of acquiring real estate due to high prices, limited inventory, and rapid market changes. Without clear insights into which property features most significantly impact pricing, buyers often find themselves either overpaying for properties or missing out on undervalued opportunities.

Traditional real estate advice relies heavily on the common mantra of "location, location, location," but in today's data-driven world, we can dig deeper. Property values are influenced by a complex combination of factors including location characteristics (zip code), physical attributes (square footage, number of bedrooms, building age), property condition, and amenities (parking, heating systems, fireplaces).

## 🎯 Goal
This project aims to demystify Boston's residential real estate market by analyzing the 2025 Property Assessment dataset to identify which features have the strongest influence on property values. By leveraging machine learning techniques, we can provide data-driven insights that help both first-time homebuyers and real estate investors make more strategic decisions.

 This is an end-to-end Machine Learning project where the data of our interest is selected, exploratory data analysis as well as cleaning of the data is performed, at least 3 models are trained, the best model is selected, packaged, and deployed to the cloud. 

**Target Audience**:

- First-time homebuyers seeking to understand which property features offer the best value for their budget
- Real estate investors looking to identify undervalued properties with strong rental income potential
- Anyone interested in understanding the Boston residential real estate market dynamics
The ultimate goal is to build a predictive model that can accurately estimate property values based on key features, empowering users to navigate Boston's challenging real estate landscape with confidence and data-backed insights.

## 🗄️ Initial Dataset

<div style="background-color: #ffe6e6; border: 2px solid #ff4444; border-radius: 8px; padding: 15px; margin: 10px 0;">
<h4 style="color: #cc0000; margin-top: 0;">🚨 Data Processing Disclaimer</h4>
<p><strong>The initial 66-column and 183445-record dataset shown below is NOT used directly for modeling.</strong></p>
<p>This project follows a systematic approach:</p>
<ol>
<li><strong>Data Preparation Phase</strong> (<code>data_prep.ipynb</code>): Comprehensive field analysis, feature selection, and data cleaning</li>
<li><strong>Modeling Phase</strong>: EDA, Feature Engineering, and Model Training on the processed dataset</li>
</ol>
<p>Only the most relevant features that align with our problem statement will be selected for the final modeling process.</p>
<hr style="border: none; border-top: 1px solid #cc0000; margin: 10px 0;">
<p><strong>🔍 For the Detail-Oriented:</strong> If you're curious about the nitty-gritty details of the data cleaning and preparation process, dive into <code>data_prep.ipynb</code> for a comprehensive walkthrough.</p>
</div>

**Target Variable**: **💰 TOTAL_VALUE**: Total assessed value for property

**Features** <details>
<summary><strong>📋 Full Dataset Field Descriptions</strong> (Click to expand)</summary>
<br>

📖 **Official Documentation**: [View Complete Field Descriptions](https://data.boston.gov/dataset/property-assessment/resource/96b6cf8b-04f2-4b87-b78f-ba9a0ddd1573)
  <a name="luc-documentation"></a>

📖 **Official Documentation for LUC - Land Use Code**: [View Complete Breakdown of Land of Use Field Descriptions](https://data.boston.gov/dataset/property-assessment/resource/d6c1268c-cd83-4dc3-a914-bba1ed59da6d/view/84f48d02-5d3b-4533-8fb1-459119d0e2d1)

<details>
<summary><strong>🏠 Property Identification</strong></summary>

- **PID**: Unique 10‐digit parcel number. First 2 digits are the ward, digits 3 to 7 are the parcel, and digits 8 to 10 are the sub‐parcel
- **CM_ID**: 10‐digit parcel number of Condo Main, which houses all related condo units
- **GIS_ID**: Primary GIS ID
- **BLDG_SEQ**: Building sequence of parcel
- **NUM_BLDGS**: Total number of buildings of parcel

</details>

<details>
<summary><strong>📍 Location & Address</strong></summary>

- **ST_NUM**: Street number of parcel
- **ST_NAME**: Street name of parcel
- **UNIT_NUM**: Specific unit number within a housing complex
- **CITY**: City or Town of parcel
- **ZIP_CODE**: Zip code of parcel

</details>

<details>
<summary><strong>🏘️ Property Classification</strong></summary>

- **LUC**: Land Use Code
  - *[Detailed Description of Each Code - See Documentation Above](#luc-documentation)*
  - We are going to focus our attention on the Residential Property Only. The full list of the residential codes is:
    - **101**: SINGLE FAM DWELLING
    - **102**: RESIDENTIAL CONDO
    - **103**: MOBILE HOME
    - **104**: TWO-FAM DWELLING
    - **105**: THREE-FAM DWELLING
    - **106**: ADD'L RES IMPROVEMENT
    - **107**: OTHER RESIDENTIAL
    - **108**: CONDO PARKING
    - **109**: MULTIPLE BUILDINGS
    - **110**: CONDO STORAGE

- **LU**: Type of Property (land use)
  - **A**: Residential 7 or more units
  - **AH**: Agricultural/Horticultural
  - **C**: Commercial
  - **CC**: Commercial condominium
  - **CD**: Residential condominium unit
  - **CL**: Commercial land
  - **CM**: Condominium main (physical structure housing all related condo units with no assessed value)
  - **CP**: Condo parking
  - **E**: Tax‐exempt
  - **EA**: Tax‐exempt (121A)
  - **I**: Industrial
  - **R1**: Residential 1‐family
  - **R2**: Residential 2‐family
  - **R3**: Residential 3‐family
  - **R4**: Residential 4 or more family
  - **RC**: Mixed use (residential and commercial)
  - **RL**: Residential land

- **LU_DESC**: Land Use Description

</details>

<details>
<summary><strong>👤 Ownership & Mailing</strong></summary>

- **OWN_OCC**: Residential Exemption: "Y" character code indicating if owner receives residential exemption as an owner‐occupied property
<div style="background-color: #ffebee; border: 2px solid #f44336; border-radius: 5px; padding: 10px; margin: 5px 0;">
<span style="color: #d32f2f; font-weight: bold;">⚠️ PII VIOLATION WARNING:</span> 
<strong>OWNER</strong> field contains personally identifiable information and should NEVER be published publicly or used in modeling.
</div>

- **MAIL_ADDRESSEE**: Care of recipient
- **MAIL_ADDRESS**: Street address where tax bill is mailed
- **MAIL_CITY**: City/neighborhood where tax bill is mailed
- **MAIL_STATE**: State where tax bill is mailed
- **MAIL_ZIP_CODE**: Zip code where tax bill is mailed

</details>

<details>
<summary><strong>📏 Building Size & Layout</strong></summary>

- **RES_FLOOR**: Number of residential building stories
- **CD_FLOOR**: Condominium unit floor number
- **RES_UNITS**: Number of residential units in a condominium building
- **COM_UNITS**: Number of commercial units in a condominium building
- **RC_UNITS**: Number of Residential/Commercial Units in a condominium building
- **LAND_SF**: Parcel's land area in square feet (legal area)
- **GROSS_AREA**: Gross floor area
- **LIVING_AREA**: Living area square footage of the property

</details>

<details>
<summary><strong>💰 Property Values & Taxes</strong></summary>

- **LAND_VALUE**: Total assessed land value
- **BLDG_VALUE**: Total assessed building value
- **GROSS_TAX**: Tax bill amount based on total assessed value multiplied by the tax rate per thousand

</details>

<details>
<summary><strong>🏗️ Building Characteristics & Age</strong></summary>

- **BLDG_TYPE**: Building Type and Style
- **YR_BUILT**: Year property was built
- **YR_REMODEL**: Year property was last remodeled
- **STRUCTURE_CL**: Structural classification of building
  - **A**: Struct Steel
  - **B**: Reinforced Concrete
  - **C**: Brick/Concrete
  - **D**: Wood/Frame
  - **E**: Metal
  - **R**: Residential

</details>

<details>
<summary><strong>🏠 Exterior Features</strong></summary>

- **ROOF_STRUCTURE**: Roof Structure Types
  - **F**: Flat
  - **G**: Gable
  - **H**: Hip
  - **L**: Gambrel
  - **M**: Mansard
  - **O**: Other
  - **S**: Shed

- **ROOF_COVER**: Roof Cover Material
  - **A**: Asphalt Shingle
  - **C**: Composition
  - **O**: Other
  - **R**: Rubber Roof
  - **S**: Slate
  - **T**: Tile
  - **W**: Wood Shingle

- **EXT_FINISHED**: Exterior Siding Material

</details>

<details>
<summary><strong>🔧 Property Condition</strong></summary>

- **INT_WALL**: Interior Wall Condition‐Residential
- **INT_COND**: Interior Condition of parcel‐Residential
- **EXT_COND**: Exterior Condition of parcel
- **OVERALL_COND**: Overall condition of parcel
- **BDRM_COND**: Bedroom Condition‐Residential

</details>

<details>
<summary><strong>🛏️ Rooms & Living Spaces</strong></summary>

- **BED_RMS**: Total number of bedrooms‐Residential
- **FULL_BTH**: Total number of full baths‐Residential
- **HALF_BTH**: Total number of half baths‐Residential
- **KITCHEN**: Total number of kitchens‐Residential
- **TT_RMS**: Total number of rooms‐Residential

</details>

<details>
<summary><strong>🚿 Bathroom Details</strong></summary>

- **BTHRM_STYLE1**: Residential bath style ‐bathroom #1
- **BTHRM_STYLE2**: Residential bath style ‐bathroom #2
- **BTHRM_STYLE3**: Residential bath style ‐ bathroom #3

</details>

<details>
<summary><strong>🍳 Kitchen Details</strong></summary>

- **KITCHEN_TYPE**: Kitchen Type ‐ Residential
- **KITCHEN_STYLE1**: Residential kitchen style –kitchen #1
- **KITCHEN_STYLE2**: Residential kitchen style – kitchen #2
- **KITCHEN_STYLE3**: Residential kitchen style –kitchen #3

</details>

<details>
<summary><strong>🌡️ Heating & Cooling Systems</strong></summary>

- **HEAT_TYPE**: Heating type
- **HEAT_SYSTEM**: Heating System (individually controlled, common or self‐contained)
- **AC_TYPE**: Air Conditioning Type‐Residential

</details>

<details>
<summary><strong>⭐ Amenities & Special Features</strong></summary>

- **FIRE_PLACE**: Total number of fireplaces
- **NUM_PARKING**: Number of parking spaces
- **PROP_VIEW**: Property View
- **ORIENTATION**: Indicates the location a condominium unit in the building is facing
- **CORNER_UNIT**: Indicates if the unit is locating in the corner of the building (Y/N)

</details>
</details>

## 🧹 Cleaned Data

## 📊 EDA

## 🤖 Model training

## Python scripts for data pre-processing and training

## UV setup
From your root directory, run `pip install uv`

Clone the repo

In the repo directory, run:

`uv init`
`rm main.py`

Install dependencies
`uv add scikit-learn==1.1.0 numpy==1.26.1 fastapi uvicorn`

Install dev dependencies
`uv add --dev requests`

## Run the code from the UV environment

`uv run uvicorn predict:app --host 0.0.0.0 --port 9696 --reload`

## Putting Everything to Docker

Make sure you have docker installed - you probably already do since you are taking this course

### Local run option:

To double check, run 

`docker run hello-world`

If it was ran successfully, then let's take a look at the file running commands:

``
docker build --no-cache -t real-estate-prediction .
``

```
docker run -it --rm -p 9696:9696 real_estate_price_prediction
```

Don't forget to stop the containers after you are done testing locally
```
docker stop $(docker ps -q)
```
Navigate to `http://0.0.0.0:9696/docs` and click on `Try it out`.
Copy-paste re_property.json file.

Expected reponse:

```python
{
  "predicted_value": 807383.14
}
```

### Cloud deployment - video proof (No need to run the code)

execute: 

```sh
curl -L https://fly.io/install.sh | sh
```

```sh
nano ~/.zshrc
```

export variables 
```sh
export FLYCTL_INSTALL="/Users/I556249/.fly"
export PATH="$FLYCTL_INSTALL/bin:$PATH"
```

Reload the shell:

```sh
`source ~/.zshrc
```
Check everything works ok by checking the fly version
```sh
which fly
```

Authenticate to fly.io:
```fly auth signup```

```fly launch --generate-name```

Answers to questions:
N - no, I don't want to tweak the settings
Y- yes, Create a Docker file

Check that Docker ignore was created

```fly deploy```

Get the name of the deployment link

Navigate to [deployment link]/docs

Try it out!

Change the marketing.py script to the URL that was created

Run python marketing.py

Destroy the app

Get the list of apps

```sh
fly apps list
```

Detroy the app
```sh
fly apps destroy <app-name>
```
