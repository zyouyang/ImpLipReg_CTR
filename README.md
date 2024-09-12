## Scaled Supervision is an Implicit Lipschitz Regularizer
This is the anonymous code repository for the paper "**Scaled Supervision is an Implicit Lipschitz Regularizer**". It adapts from the public code repository [RecBole](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj_oLzE7ruIAxUwGFkFHan_GDIQFnoECAkQAQ&url=https%3A%2F%2Fgithub.com%2FRUCAIBox%2FRecBole&usg=AOvVaw3tePg3tzwZUWkgSKsBHBzh&opi=89978449). 

### How to Run
1. Install all the required packages found [here](https://github.com/RUCAIBox/RecBole/blob/master/requirements.txt).
2. Download all the benchmark datasets used in the paper, including ML-1M, Yelp2018, and Amazon-book. Make sure the folders are named `ml-1m`, `yelp2018`, and `amazon-book` respectively.
3. Run the command `python run_recbole.py --model=<model_name> --dataset=<dataset_name> [hyper_parameter_setting]` to train a new CTR model from scratch.
4. To apply our approach, add the hyperparameter settings `--multi_cls=5 --multi_ratio=<chosen_ratio>`. You may consult the commands in `commands.sh`.
5. To evaluate the ranking metrics, `python recbole_rank_eval.py --checkpoint=<saved_model_checkpoint>`.

### Acknowledgement
This repository adapts from the public code repository [RecBole](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj_oLzE7ruIAxUwGFkFHan_GDIQFnoECAkQAQ&url=https%3A%2F%2Fgithub.com%2FRUCAIBox%2FRecBole&usg=AOvVaw3tePg3tzwZUWkgSKsBHBzh&opi=89978449). We remain the author names of [RecBole](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj_oLzE7ruIAxUwGFkFHan_GDIQFnoECAkQAQ&url=https%3A%2F%2Fgithub.com%2FRUCAIBox%2FRecBole&usg=AOvVaw3tePg3tzwZUWkgSKsBHBzh&opi=89978449) in the corresponding files as they are publicly. Any name appear in the repository is irrelavent to the authorship of this work. This code repository is strictly anonymous to the work "**Scaled Supervision is an Implicit Lipschitz Regularizer**".