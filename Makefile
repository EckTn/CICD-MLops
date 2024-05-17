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
#	echo "![Confusion Matrix](./results/confusionmatrix.png)" >> report.md

	# Encoding image to base64
#	BASE64_IMG=$(base64 -w 0 ./results/confusionmatrix.png)
	IMAGE_URL=$(cml publish ./results/confusionmatrix.png)


	# Debbuging
	if [ -f "./results/confusionmatrix.png" ]; then \
#	    echo "![](./results/confusionmatrix.png)" >> report.md; \
#	    echo '<img src="data:image/png;base64,'$BASE64_IMAGE'" alt="Confusion Matrix"/>' >> report.md; \
	    echo "![Confusion Matrix]($IMAGE_URL)" >> report.md; \
	else \
	    echo "Confusion matrix image not found." >> report.md; \
	fi

	ls -lh ./results/confusionmatrix.png || echo "Confusion matrix image not found."
	cat report.md
	ls -lh report.md

	cml comment create report.md

#update-branch:
#	update-branch:
#	git config --global user.name $(USER_NAME)
#	git config --global user.email $(USER_EMAIL)
#	git commit -am "Update with new results"
#	git push --force origin HEAD:update
