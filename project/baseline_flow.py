from metaflow import FlowSpec, step, Flow, current, Parameter, IncludeFile, card, current
from metaflow.cards import Table, Markdown, Artifact
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from typing import Tuple
import pandas as pd

def get_clf(
    ngram_range: Tuple[int, int] = (1, 3), max_iter: int = 1000, random_seed: int = 42
) -> Pipeline:
    classifier = LogisticRegression(max_iter=max_iter, random_state=random_seed)

    char_vectorizer = TfidfVectorizer(
        min_df=0.001,
        max_df=0.9,
        ngram_range=ngram_range,
        use_idf=True,
        norm="l2",
        analyzer="char_wb",
        smooth_idf=True,
        strip_accents="unicode",
        sublinear_tf=True,
        lowercase=True,
    )

    word_vectorizer = TfidfVectorizer(
        min_df=0.001,
        max_df=0.9,
        ngram_range=ngram_range,
        use_idf=True,
        norm="l2",
        analyzer="word",
        smooth_idf=True,
        strip_accents="unicode",
        sublinear_tf=True,
        lowercase=True,
    )

    features = FeatureUnion([("chars", char_vectorizer), ("words", word_vectorizer)])
    classifier_pipeline = make_pipeline(features, classifier)
    return classifier_pipeline

# TODO move your labeling function from earlier in the notebook here
def labeling_function(row: pd.Series, rating_column: str = "rating", positive_threshold: int = 3) -> int:
    return 1 if row[rating_column] > positive_threshold else 0


class BaselineNLPFlow(FlowSpec):

    # We can define input parameters to a Flow using Parameters
    # More info can be found here https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows
    split_size = Parameter('split-sz', default=0.2)
    # In order to use a file as an input parameter for a particular Flow we can use IncludeFile
    # More information can be found here https://docs.metaflow.org/api/flowspec#includefile
    data = IncludeFile('data', default='../data/Womens Clothing E-Commerce Reviews.csv')

    @step
    def start(self):

        # Step-level dependencies are loaded within a Step, instead of loading them 
        # from the top of the file. This helps us isolate dependencies in a tight scope.
        import pandas as pd
        import io 
        from sklearn.model_selection import train_test_split
        
        # load dataset packaged with the flow.
        # this technique is convenient when working with small datasets that need to move to remove tasks.
        df = pd.read_csv(io.StringIO(self.data))

        # filter down to reviews and labels 
        df.columns = ["_".join(name.lower().strip().split()) for name in df.columns]
        df['review_text'] = df['review_text'].astype('str')
        _has_review_df = df[df['review_text'] != 'nan']
        reviews = _has_review_df['review_text']
        labels = _has_review_df.apply(labeling_function, axis=1)
        # Storing the Dataframe as an instance variable of the class
        # allows us to share it across all Steps
        # self.df is referred to as a Data Artifact now
        # You can read more about it here https://docs.metaflow.org/metaflow/basics#artifacts
        self.df = pd.DataFrame({'label': labels, **_has_review_df})
        del df
        del _has_review_df

        # split the data 80/20, or by using the flow's split-sz CLI argument
        _df = pd.DataFrame({'review': reviews, 'label': labels})
        self.traindf, self.valdf = train_test_split(_df, test_size=self.split_size)
        print(f'num of rows in train set: {self.traindf.shape[0]}')
        print(f'num of rows in validation set: {self.valdf.shape[0]}')

        self.next(self.baseline)

    @step
    def baseline(self):
        "Compute the baseline"
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        ### TODO: Fit and score a baseline model on the data, log the acc and rocauc as artifacts.
        clf = get_clf()
        clf.fit(self.traindf["review"], self.traindf["label"])
        preds = clf.predict(self.valdf["review"])
        actuals = self.valdf["label"]
        self.valdf["prediction"] = preds
        self.base_acc = accuracy_score(actuals, preds)
        self.base_rocauc = roc_auc_score(actuals, preds)

        self.next(self.end)
        
    @card(type='corise') # TODO: after you get the flow working, chain link on the left side nav to open your card!
    @step
    def end(self):

        def get_examples(df: pd.DataFrame, looking_for_label: int, looking_for_prediction: int, label_column: str = "label", prediction_column: str = "prediction", review_column: str = "review") -> Tuple[int, str]:
            examples_df = df[(df[label_column] == looking_for_label) & (df[prediction_column] == looking_for_prediction)]
            examples = (examples_df[review_column].str.replace("\n", " ")).sample(10, random_state=42).tolist()
            examples = "".join([f"* {e}\n" for e in examples])
            return examples_df.shape[0], examples


        msg = 'Baseline Accuracy: {}\nBaseline AUC: {}'
        print(msg.format(
            round(self.base_acc,3), round(self.base_rocauc,3)
        ))

        current.card.append(Markdown("# Womens Clothing Review Results"))
        current.card.append(Markdown("## Overall Accuracy"))
        current.card.append(Artifact(self.base_acc))

        current.card.append(Markdown("## Examples of False Positives"))

        # TODO: compute the false positive predictions where the baseline is 1 and the valdf label is 0. 
        # TODO: display the false_positives dataframe using metaflow.cards
        # Documentation: https://docs.metaflow.org/api/cards#table
        num_false_positives, examples = get_examples(self.valdf, 0, 1)
        current.card.append(Markdown(f"Total number of false positives, i.e., when the true label is NEGATIVE (0), but the classifier predicted POSITIVE (1): {num_false_positives}"))
        current.card.append(Markdown(f"Examples:\n{examples}"))
        
        current.card.append(Markdown("## Examples of False Negatives"))
        # TODO: compute the false positive predictions where the baseline is 0 and the valdf label is 1. 
        # TODO: display the false_negatives dataframe using metaflow.cards
        false_negatives_df = self.valdf[(self.valdf["label"] == 1) & (self.valdf["prediction"] == 0)]
        current.card.append(Markdown(f"Total number of false negatives, i.e., when the true label is POSITIVE (1), but the classifier predicted NEGATIVE (0): {false_negatives_df.shape[0]}"))
        false_negatives = false_negatives_df["review"].sample(10, random_state=42).tolist()
        false_negatives = "".join([f"* {fn}\n" for fn in false_negatives])
        current.card.append(Markdown(f"Examples:\n{false_negatives}"))


if __name__ == '__main__':
    BaselineNLPFlow()
