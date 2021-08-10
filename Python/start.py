from flask import Flask,abort,render_template,request
import OO_CODE

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST','GET'])
def result():
    if request.method == 'POST':
        filename = request.form['dataset_file']
        result =OO_CODE.full_proj(filename)
    return render_template('result.html',result= result)

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')
	
@app.route('/check', methods=['POST','GET'])
def check():
    return render_template('fields.html')

@app.route('/checkvalue', methods=['POST','GET'])
def checkvalue():
	if request.method == 'POST':
		filename = request.form['dataset_file']
		if filename == "Heart2.csv":
				age = request.form['x1']
				sex = request.form['x2']
				alcohol = request.form['x3']
				education = request.form['x4']
				famhist = request.form['x5']
				typea = request.form['x6']
				obesity = request.form['x7']
				ldl = request.form['x8']
				attacked = request.form['x9']
				sbp = request.form['x10']
				dbp = request.form['x11']
				pulse = request.form['x12']
				glucose = request.form['x13']
				adiposity = request.form['x14']
				tobacco = request.form['x15']
				values = [int(age),float(sex),float(alcohol),float(education),float(famhist),float(typea),float(obesity),float(ldl),float(attacked),float(sbp),float(dbp),float(pulse),float(glucose),float(adiposity),float(tobacco)]

		else:
				Gender = request.form['x1']
				Age = request.form['x2']
				Education = request.form['x3']
				CurrentSmoker = request.form['x4']
				CigsPerDay = request.form['x5']
				BPMeds = request.form['x6']
				PrevalentStroke = request.form['x7']
				PrevalentHyp = request.form['x8']
				Diabetes = request.form['x9']
				TotChol = request.form['x10']
				SysBP = request.form['x11']
				DiaBP = request.form['x12']
				BMI= request.form['x13']
				HeartRate= request.form['x14']
				Glucose = request.form['x15']
				values = [int(Gender),int(Age),int(Education),int(CurrentSmoker),int(CigsPerDay),int(BPMeds),int(PrevalentStroke),int(PrevalentHyp),float(Diabetes),float(TotChol),float(SysBP),float(DiaBP),float(BMI),float(HeartRate),float(Glucose)]

		checkvalue = OO_CODE.check(values,filename)
		return render_template('final.html',checkvalue=checkvalue)

if __name__ == '__main__':
	app.run(debug = False)
