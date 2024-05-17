install:
	pip	install --upgrade pip &&\
    pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./results/metrics.txt >> report.md
	echo "" >> report.md
	echo "## Confusion Matrix Plot" >> report.md
	echo "![Confusion Matrix](./results/confusionmatrix.png)" >> report.md

#	# Debbuging
#	if [ -f "./results/test_img.png" ]; then \
#	    echo "![Confusion Matrix](./results/test_img.png)" >> report.md; \
#	else \
#	    echo "Confusion matrix image not found." >> report.md; \
#	fi

#	ls -lh ./results/confusion_matrix.png || echo "Confusion matrix image not found."
#	cat report.md
#	ls -lh report.md

	cml comment create report.md