

class sample_triplets:
    
    def __init__(self, df_query_data):
        # The data frame training data used fopr the model
        self.df_query_data = df_query_data
        
    def sample_batch(self, batch_size):
        df_batch_sample = self.df_query_data.sample(n=batch_size)
        image_triplets = list()
        annotation_triplets = list()

        for index, row in df_batch_sample.iterrows():
            # Create the annotation for image by concatinating, query, param1 and param2
            ann = row['annotations']
            ann = ann.lower()
            
            # Create image name from external_image_id number
            img_id = row['image_name']
            img_name = '{}'.format(img_id)
            
            # Sample negative image for pair, by sampling a random item, not corresponding to the same param2 cat
            neg_sample = self.df_query_data.sample(n=2)
            neg_img_name = neg_sample['image_name'].iloc[0]
             # Sample negative annotation for pair, by sampling a random item, not corresponding to the same param2 cat
            neg_ann = neg_sample['annotations'].iloc[1]
            neg_ann = neg_ann.lower()
            
            # Construct and append the new triplet samples
            image_triplets.append((img_name, ann, neg_ann))
            annotation_triplets.append((ann, img_name, neg_img_name))

        return image_triplets, annotation_triplets
    
    
    