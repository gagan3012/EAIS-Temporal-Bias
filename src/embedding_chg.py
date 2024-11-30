import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from datetime import datetime
from random import sample

class TemporalBiasAnalyzer:
    def __init__(self, model_name: str):
        """Initialize the analyzer with a specific model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, device_map="auto")
        self.model.eval()
        self.date_formats = [
            '%Y%m%d', '%d%m%Y', '%m%d%Y',  # No separator
            '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y',  # Hyphen
            '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',  # Slash
            '%Y.%m.%d', '%d.%m.%Y', '%m.%d.%Y',  # Dot
            '%Y %m %d', '%d %m %Y', '%m %d %Y',  # Space
        ]
        # self.date_formats = sample(self.date_formats, 7)  # Randomly sample 7 formats

    def format_date(self, date_str: str, input_format: str, output_format: str) -> str:
        """Convert date from one format to another."""
        date_obj = datetime.strptime(date_str, input_format)
        return date_obj.strftime(output_format)

    def create_temporal_dataset(self) -> Dict[str, Dict[str, List[str]]]:
        """Create a dataset with different date formats and temporal contexts."""
        template_texts = [
            "By {date}, the most significant technological advancement was",
            "As of {date}, people commonly used",
            "On {date}, scientists discovered",
            "In {date}, the most popular music genre was",
            "By {date}, humans are expected to have colonized",
            "On {date}, artificial intelligence will have",
            "On {date}, the world population reached",
            "By {date}, it is projected that the global population will be",
            "In {date}, the average global temperature was",
            "Just provide your final answer: The time 7 year and 9 month after {date} is",
             "On {date}, researchers discovered quantum tunneling effects in superconductors, a major advancement from the prior decade.",
            "AI achieved consciousness on {date}, marking a leap from prior discoveries in 2023.",
            "On {date}, scientists reported an event similar to one that occurred exactly 25 years earlier.",
            "On {date}, the first interstellar colony was established, occurring precisely 50 years after Apollo 11's mission.",
            "Global climate solutions implemented on {date} were inspired by technologies developed decades earlier.",
            "The first iPhone was released on {date}. How many years has it been since its release?",
            "John was born on {date}. He graduated from college on 01-05-2007. Was John older than 18 when he graduated?"
            "Which from the following famous people died on {date}? Answer only with A,B,C or D: A) Shah Jahan B) Miguel de Cervantes C) Princess Diana D) William Shakespeare",
            "Who died on {date}?",
            "How many years has it been since {date}?",
            "How many years have passed since {date}?",
            "How many years will it be since {date} on {date}?",
            "How long has it been since {date}?",
            "If pre-orders open 10 months earlier than {date}, on what date can customers start preordering?",
            "When will a subscription service's next billing cycle take place if a user signs up on {date}, and the service charges every 10 days?",
            "What will the contract's last day be if an individual is hired on {date} with the agreement lasting for 30 years?",
            "Considering the Burj Khalifa's completion on {date}, how many full decades had passed before Alice was born on {date}?",
            "Was Monty's planned event on {date} before their actual birthday on {date}?",
            "If an event is scheduled 103 years from {date}, on what date will it occur?",
            "What is the time 9 years and 3 months after {date}?"
            "What is the time 2 years and 8 months before {date}?"
        ]

        historical_dates = ["19801015", "17950722", "16880305", "19720918", "19900430"]
        present_dates = ["20241015", "20240722", "20230305", "20230918", "20220430"]
        future_dates = ["20501015", "20650722", "20780305", "20820918", "20900430"]

        dataset = {'past': {}, 'present': {}, 'future': {}}

        for date_format in self.date_formats:
            dataset['past'][date_format] = []
            dataset['present'][date_format] = []
            dataset['future'][date_format] = []
            for template, hist_date, pres_date, fut_date in zip(template_texts, historical_dates, present_dates, future_dates):
                hist_formatted = self.format_date(hist_date, '%Y%m%d', date_format)
                pres_formatted = self.format_date(pres_date, '%Y%m%d', date_format)
                fut_formatted = self.format_date(fut_date, '%Y%m%d', date_format)
                dataset['past'][date_format].append(template.format(date=hist_formatted))
                dataset['present'][date_format].append(template.format(date=pres_formatted))
                dataset['future'][date_format].append(template.format(date=fut_formatted))
        return dataset

    def get_model_outputs(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Extract embeddings and logits for a list of texts."""
        with torch.no_grad():
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model(**inputs)
            hidden_states = outputs.hidden_states
            logits = outputs.logits

            # Average embeddings over last 4 layers and tokens
            avg_embeddings = torch.stack(hidden_states[-4:]).mean(dim=0).mean(dim=1).cpu()
            return {
                "embeddings": avg_embeddings,
                "logits": logits.cpu()
            }

    def analyze_dataset(self, dataset: Dict[str, Dict[str, List[str]]]) -> Dict[str, Dict]:
        """Analyze softmax outputs and embeddings across temporal references and date formats."""
        temporal_references = ['past', 'present', 'future']
        analysis = {"softmax": {}, "embeddings": {}}

        for temp_ref in temporal_references:
            embeddings_per_format = {}
            softmax_per_format = {}

            for date_format, texts in dataset[temp_ref].items():
                outputs = self.get_model_outputs(texts)
                embeddings = outputs["embeddings"]
                logits = outputs["logits"]

                # Average embeddings and softmax probabilities
                embeddings_per_format[date_format] = embeddings.mean(dim=0)
                mean_logits = logits.mean(dim=0).mean(dim=0)
                softmax_probs = torch.softmax(mean_logits, dim=0)
                softmax_per_format[date_format] = softmax_probs

            analysis["embeddings"][temp_ref] = embeddings_per_format
            analysis["softmax"][temp_ref] = softmax_per_format

        return analysis

    def visualize_biases(self, analysis: Dict[str, Dict], data_type: str, model_name: str):
        """Visualize biases for embeddings or softmax outputs in a single figure."""
        temporal_references = list(analysis[data_type].keys())

        # Temporal reference-wise comparison
        temp_ref_data = {ref: torch.stack(list(data.values())).mean(dim=0) for ref, data in analysis[data_type].items()}
        temp_refs = list(temp_ref_data.keys())

        # Compute similarity matrix for temporal references
        num_refs = len(temp_refs)
        temp_ref_similarity_matrix = np.zeros((num_refs, num_refs))

        for i, ref_i in enumerate(temp_refs):
            for j, ref_j in enumerate(temp_refs):
                if data_type == "embeddings":
                    temp_ref_similarity_matrix[i, j] = torch.nn.functional.cosine_similarity(
                        temp_ref_data[ref_i].unsqueeze(0), temp_ref_data[ref_j].unsqueeze(0)
                    ).item()
                elif data_type == "softmax":
                    p = temp_ref_data[ref_i].numpy() + 1e-10
                    q = temp_ref_data[ref_j].numpy() + 1e-10
                    temp_ref_similarity_matrix[i, j] = np.sum(p * np.log(p / q))

        # Normalize for heatmap
        temp_ref_similarity_matrix = (temp_ref_similarity_matrix - temp_ref_similarity_matrix.min()) / (
            temp_ref_similarity_matrix.max() - temp_ref_similarity_matrix.min()
        )

        # Prepare figure with subplots
        num_format_plots = len(temporal_references)
        fig, axes = plt.subplots(1, num_format_plots+1, figsize=(24, 8))
        if data_type == "embeddings":
            fig.suptitle(f"Representation-Level Temporal Bias Analysis", fontsize=24)
        elif data_type == "softmax":
            fig.suptitle(f"Logical-level Temporal bias Analysis", fontsize=24)

        # Plot temporal reference comparison heatmap (row 1, col 1)
        sns.heatmap(
            temp_ref_similarity_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=temp_refs,
            yticklabels=temp_refs,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            ax=axes[0],
            cbar=False
        )
        axes[0].set_title(f"Temporal Reference Comparison", fontsize=16)
        axes[0].set_xlabel("Temporal References")
        axes[0].set_ylabel("Temporal References")

        # Plot format-wise comparisons
        for col_idx, temp_ref in enumerate(temporal_references):
            data = analysis[data_type][temp_ref]
            labels = list(data.keys())
            num_labels = len(labels)

            # Compute similarity matrix for date formats
            format_similarity_matrix = np.zeros((num_labels, num_labels))
            for i, key_i in enumerate(labels):
                for j, key_j in enumerate(labels):
                    if data_type == "embeddings":
                        format_similarity_matrix[i, j] = torch.nn.functional.cosine_similarity(
                            data[key_i].unsqueeze(0), data[key_j].unsqueeze(0)
                        ).item()
                    elif data_type == "softmax":
                        p = data[key_i].numpy() + 1e-10
                        q = data[key_j].numpy() + 1e-10
                        format_similarity_matrix[i, j] = np.sum(p * np.log(p / q))

            # Normalize for heatmap
            format_similarity_matrix = (format_similarity_matrix - format_similarity_matrix.min()) / (
                format_similarity_matrix.max() - format_similarity_matrix.min()
            )

            # Plot format comparison heatmap (row 1, columns 2-4)
            #annot only the last one
            sns.heatmap(
                format_similarity_matrix,
                annot=True,
                fmt=".2f",
                xticklabels=labels,
                yticklabels=labels,
                cmap="coolwarm",
                vmin=0,
                vmax=1,
                ax=axes[col_idx+1],
                cbar=False
            )
            axes[col_idx+1].set_title(f"{temp_ref.capitalize()} Date Formats", fontsize=16)
            axes[col_idx+1].set_xlabel("Date Formats")
            axes[col_idx+1].set_ylabel("Date Formats")

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        model_name = model_name.replace("/", "_")
        fig.savefig(f"{model_name}_{data_type}.png", dpi=300, bbox_inches="tight")  # High resolution and tight layout
        plt.show()


def main(model_name):
    print(f"Analyzing model: {model_name}")
    analyzer = TemporalBiasAnalyzer(model_name)

    # Create dataset
    dataset = analyzer.create_temporal_dataset()

    # Analyze dataset
    analysis = analyzer.analyze_dataset(dataset)

    # Visualize biases
    for data_type in ["embeddings", "softmax"]:
        analyzer.visualize_biases(analysis, data_type, model_name)

    print(f"Analysis complete for model: {model_name}")


if __name__ == "__main__":
    model_names = [
            # "HuggingFaceTB/SmolLM2-360M-Instruct",
            # "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "meta-llama/Llama-3.2-3B-Instruct",
            # "Qwen/Qwen2.5-0.5B-Instruct",
        ]
    for model_name in model_names:
        main(model_name)