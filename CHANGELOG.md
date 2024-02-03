# CHANGELOG (3rd Feb 2024)

## System configuration

* Hardware - Apple M2 Pro Chip
* OS - MacOS 14.3
* Python version - v3.8.18
* Pip - v23.0.1

## Package Versions

1. numpy==1.24.3
2. scikit-learn==1.0.2
3. scikit-image==0.19.2
4. pandas==2.0.3
5. tensorflow==2.13.0
6. tensorflow-metal - Optional and only for Apple silicon
7. xgboost==2.0.3
8. astropy==5.2.2
9. matplotlib==3.6.2

## Code Change

1. CMETNet_CNN_train.py - Line 174

    **Old Code**

    `final_CNN_results = pred_results.append(events_with_no_images).sort_values(by=['event_number'])`

    **New Code**

    `final_CNN_results = pd.concat([pred_results, events_with_no_images]).sort_values(by=['event_number'])`
