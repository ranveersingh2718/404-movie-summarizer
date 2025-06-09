import pandas as pd
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class MovieSearchWithSummarizer:
    def __init__(self, csv_path="wiki_movie_plots_deduped.csv", summary_sentences=3):
        self.csv_path = csv_path
        self.df = None
        self.gpt2_model = None
        self.gpt2_tokenizer = None
        self.summary_sentences = summary_sentences
        
        self.load_data()
        self.load_gpt2()
    
    def load_data(self):
        # check csv exists
        if not os.path.exists(self.csv_path):
            print(f"Error: {self.csv_path} not found in current directory.")
            print("Please make sure the file is in the same directory as this script.")
            exit()
        
        # load CSV into pandas DataFrame
        print(f"Loading {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path, encoding='utf-8')
        print(f"Loaded {len(self.df)} movies successfully!")
    
    def load_gpt2(self):
        # load GPT-2 model and tokenizer
        print("Initializing GPT-2 for plot summarization...")
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        self.gpt2_tokenizer.pad_token = self.gpt2_tokenizer.eos_token
        print("GPT-2 loaded successfully!")
    
    def summarize_plot(self, plot_text):
        # create prompt
        prompt = f"""Write a narrative movie plot summary in exactly {self.summary_sentences} sentences. Use third person narration and describe the key events in order:

        {plot_text[:512]}

        Plot Summary ({self.summary_sentences} sentences):"""
        
        # tokenize input text
        inputs = self.gpt2_tokenizer(prompt, return_tensors="pt", max_length=600, truncation=True)
        
        # calculate output length parameters based on desired sentence count
        tokens_per_sentence = 25
        max_length_add = tokens_per_sentence * self.summary_sentences + 40
        min_length_add = tokens_per_sentence * (self.summary_sentences - 1)
        
        # generate text using GPT-2
        with torch.no_grad():
            outputs = self.gpt2_model.generate(
                **inputs,  # tokenized input prompt
                max_length=inputs.input_ids.shape[-1] + max_length_add,  # max total output length in tokens
                min_length=inputs.input_ids.shape[-1] + min_length_add,  # min total output length in tokens
                do_sample=True,  # enable probabilistic sampling
                temperature=0.8,  # controls randomness (0.8 balances creativity and coherence)
                top_k=40,  # sample from top 40 most probable tokens
                top_p=0.9,  # only consider tokens in top 90% probability
                no_repeat_ngram_size=3,  # prevent repeating any 3-word sequences
                pad_token_id=self.gpt2_tokenizer.eos_token_id
            )
        
        # convert generated tokens back to text
        full_response = self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # format the generated summary
        try:
            parts = full_response.split(f"Plot Summary ({self.summary_sentences} sentences):")
            
            if len(parts) > 1:
                summary = parts[1].strip()
            else:
                # try other delimiters as fallback
                for sep in ["\n\n", "\n", "Summary:", "Plot:"]:
                    if sep in full_response:
                        summary = full_response.split(sep, 1)[1].strip()
                        break
                else:
                    summary = full_response.strip()
            
            # process sentences, ensure proper formatting
            sentences = [s.strip() for s in summary.split('.') if s.strip()]
            sentences = [s + '.' if not s.endswith('.') else s for s in sentences]
            
            if len(sentences) >= self.summary_sentences:
                final_summary = ' '.join(sentences[:self.summary_sentences])
            elif len(sentences) > 0:
                final_summary = ' '.join(sentences)
            else:
                final_summary = self.create_basic_summary(plot_text)
            
            return final_summary
            
        except Exception as e:
            print(f"Error in summarization: {e}")
            return self.create_basic_summary(plot_text)
    
    def create_basic_summary(self, plot_text):
        sentences = [s.strip() for s in plot_text.split('.') if s.strip()]
        
        if len(sentences) >= self.summary_sentences:
            return '. '.join(sentences[:self.summary_sentences]) + '.'
        elif len(sentences) > 0:
            return '. '.join(sentences) + '.'
        else:
            return "Error creating basic summary"
    
    def search_movies(self, search_term):
        if self.df is None:
            print("Data not loaded.")
            return []
        
        # case-insensitive title search
        results = self.df[self.df['Title'].str.contains(search_term, case=False, na=False)]
        return results
    
    def display_search_results(self, results):
        if results.empty:
            print("\nNo movies found matching your search.")
            return False
        
        print(f"\nFound {len(results)} movie(s):")
        print("-" * 40)
        
        # show numbered list for selection
        for idx, (_, movie) in enumerate(results.iterrows(), 1):
            print(f"{idx}. {movie['Title']} ({movie['Release Year']})")
        
        print("-" * 40)
        return True
    
    def display_movie_details(self, movie):
        # display comprehensive movie information wiyh formatting
        print("\n" + "=" * 80)
        print(f"Movie Details: {movie['Title']} ({movie['Release Year']})")
        print("=" * 80)
        print(f"Director: {movie['Director']}")
        print(f"Cast: {movie['Cast']}")
        print(f"Genre: {movie['Genre']}")
        print(f"Origin/Ethnicity: {movie['Origin/Ethnicity']}")
        print(f"Wiki Page: {movie['Wiki Page']}")
        
        # generate and display summary
        print(f"\nGenerating {self.summary_sentences}-sentence plot summary...")
        summary = self.summarize_plot(movie['Plot'])
        print("\nPlot Summary:")
        print("-" * 80)
        print(summary)
        print("-" * 80)
        
        # option to view full plot if desired
        show_full = input("\nWould you like to see the full plot? (y/n): ").lower().strip()
        if show_full == 'y':
            print("\nFull Plot:")
            print("-" * 80)
            print(movie['Plot'])
            print("=" * 80)
        
        print("\n" + "=" * 50)
        print()
    
    def run_search_interface(self):
        # main loop 
        print("\n***** Movie Search Application with GPT-2 Summarizer *****")
        #print(f"Database contains {len(self.df)} movies from {self.df['Release Year'].min()} to {self.df['Release Year'].max()}")
        
        while True:
            print("\n" + "-" * 50)
            search_term = input("Enter movie title to search (or 'q' to quit): ").strip()
            
            # handle quit commands
            if search_term.lower() in ['q', 'quit', 'exit']:
                print("\nQuitting.")
                break
            
            if not search_term:
                print("Please enter a valid search term.")
                continue
            
            # execute search and display results
            results = self.search_movies(search_term)
            has_results = self.display_search_results(results)
            
            # allow movie selection if results found
            if has_results:
                try:
                    selection = input(f"\nEnter movie number (1-{len(results)}) to view details, or Enter to search again: ").strip()
                    
                    if selection:
                        selection_idx = int(selection) - 1
                        if 0 <= selection_idx < len(results):
                            selected_movie = results.iloc[selection_idx]
                            self.display_movie_details(selected_movie)
                        else:
                            print("Invalid selection. Please try again.")
                            
                except ValueError:
                    print("Invalid input. Please enter a number.")

# create and run application
if __name__ == "__main__":
    search_engine = MovieSearchWithSummarizer(summary_sentences=3)
    search_engine.run_search_interface()