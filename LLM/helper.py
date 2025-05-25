# keep the notebook outputs intact while pushing to github

jupyter nbconvert --ClearMetadataPreprocessor.enabled=True --ClearOutputPreprocessor.enabled=False --to notebook --inplace <notebook-name>.ipynb
