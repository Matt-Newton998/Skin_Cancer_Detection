{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Melanoma Cancer Classification Using A Convolutional Neural Network\n",
    "\n",
    "The goal of this project was to classify each type of the 7 types of skin lesion found in the [ISIC Dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T). However, due to the low number of images provided of some of the types, this was switched to classifying weather the lesion was melanoma or not, as a proof of concept. This is still a very valuable tool. Melanomaic lesion types (melanoma & melanocytic) are only lesion types in the data set that can be fatal. So detecting these types compared to the others can inform a patient weather further treatment is needed. \n",
    "\n",
    "There are only [6,123 histologists](https://www.healthcareers.nhs.uk/explore-roles/healthcare-science/roles-healthcare-science/life-sciences/histopathology-healthcare-scientist) registered in the UK. With a rate of 44 new cases in the UK every day a tool to help in their work would be invaluable. Additionally, this tool could be deployed globally. Helping countries which don't have enough histologists and the rates of skin cancer are even higher.  \n",
    "\n",
    "## About Melanoma Cancer\n",
    "\n",
    "Skin cancer is the major public health issue, there are two main types of skin cancer: Non melanoma skin cancer (which includes basal cell skin cancer, squamous cell skin cancer and other rare types) and melanoma skin cancer. \n",
    "\n",
    "Melanoma skin cancer is the 5th most common cancer in the UK, accounting for 5% of all new cancer cases (2016). There are around [16,000](https://www.cancerresearchuk.org/health-professional/cancer-statistics/statistics-by-cancer-type/melanoma-skin-cancer#heading-Zero) new cases in the UK every year. That's 44 diagnoses every day! \n",
    "\n",
    "Melanoma is most common in elderly population with the median age of diagnosis being 70-75.  The rate are on the rise with the UK having an increase of 134% since 1990. However, the highest rates globally are New Zealand and Australia, with 36.5 & 33.3 cases per 100,000 people respectively.\n",
    "\n",
    "[According to the WHO in 2015](http://www-dep.iarc.fr/WHOdb/WHOdb.htm), the global incidence of melanoma was estimated to be over 350,000 cases, with almost 60,000 deaths. Although the mortality is significant, when detected early, melanoma survival exceeds 95%. Therefore, early detection is paramount.\n",
    "\n",
    "## The Dataset\n",
    "\n",
    "The dataset consists of 10015 dermatoscopic images, Provided by the International Skin Imaging Collaboration (ISIC). The ISIC is an international effort to improve melanoma diagnosis, sponsored by the International Society for Digital Imaging of the Skin (ISDIS). The ISIC Archive contains the largest publicly available collection of quality controlled dermoscopic images of skin lesions.\n",
    "\n",
    "Cases include a representative collection of all important diagnostic categories in the realm of pigmented lesions:\n",
    "- `akiec` - Actinic keratoses and intraepithelial carcinoma / Bowen's disease \n",
    "- `bcc` - basal cell carcinoma\n",
    "- `bkl` - benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses)\n",
    "- `df` - dermatofibroma\n",
    "- `mel` - melanoma\n",
    "- `nv` - melanocytic  \n",
    "- `vasc` - vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage)\n",
    "\n",
    "More than 50% of lesions are confirmed through histopathology (histo), the ground truth for the rest of the cases is either follow-up examination (follow_up), expert consensus (consensus), or confirmation by in-vivo confocal microscopy (confocal). The dataset includes lesions with multiple images, which can be tracked by the lesion_id-column within the HAM10000_metadata file.\n",
    "\n",
    "# Results\n",
    "\n",
    "The model was moderately successful in that pre-threshold selection had a precision of 0.89 and a recall of 0.89 as well. This was further improved by adjusting the threshold to give a recall of 0.99, reducing the precision to 0.79. This is chosen as it is preferable to catch as many melanoma cases as possible for the histologist to review later.    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  },
  "toc": {
   "base_numbering": 1,
   "nav_menu": {},
   "number_sections": true,
   "sideBar": true,
   "skip_h1_title": false,
   "title_cell": "Table of Contents",
   "title_sidebar": "Contents",
   "toc_cell": false,
   "toc_position": {},
   "toc_section_display": true,
   "toc_window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
