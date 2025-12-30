import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.video import r3d_18
# import torchvision.models as models

class SimpleConcatFusion(nn.Module):
    def __init__(self, image_feat_dim, video_feat_dim, hidden_dim, num_classes=10):
        super(SimpleConcatFusion, self).__init__()
        # Fusion layer (concatenate the features from both modalities)
        self.fc = nn.Sequential(
            nn.Linear(image_feat_dim + video_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image_features, video_features):
        # Concatenate image and video features
        combined_features = torch.cat((image_features, video_features), dim=1)
        # Pass through fully connected layers for classification
        out = self.fc(combined_features)
        return out


class GatedFusion(nn.Module):
    def __init__(self, image_feat_dim, video_feat_dim, hidden_dim, num_classes=10):
        super(GatedFusion, self).__init__()

        # Learnable gates for weighting image and video features
        self.gate_image = nn.Sequential(
            nn.Linear(image_feat_dim, hidden_dim),
            nn.Sigmoid()  # Sigmoid to scale values between 0 and 1
        )
        self.gate_video = nn.Sequential(
            nn.Linear(video_feat_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image_features, video_features):
        # Apply gates to each modality
        gated_image = self.gate_image(image_features) * image_features
        gated_video = self.gate_video(video_features) * video_features

        # Concatenate gated features
        combined_features = torch.cat((gated_image, gated_video), dim=1)
        
        # Pass through fully connected layers
        out = self.fc(combined_features)
        return out

class MeanFusion(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_classes=10):
        super(MeanFusion, self).__init__()
        # Fusion layer (average the features from both modalities)
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image_features, video_features):
        # Take the average of image and video features
        combined_features = (image_features + video_features) / 2
        # Pass through fully connected layers
        out = self.fc(combined_features)
        return out


class MaxPoolingFusion(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_classes=10):
        super(MaxPoolingFusion, self).__init__()
        # Fusion layer (max pooling the features from both modalities)
        self.fc = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image_features, video_features):
        # Take the element-wise max of image and video features
        combined_features = torch.max(image_features, video_features)
        # Pass through fully connected layers
        out = self.fc(combined_features)
        return out


class BilinearFusion(nn.Module):
    def __init__(self, image_feat_dim, video_feat_dim, hidden_dim, num_classes=10):
        super(BilinearFusion, self).__init__()
        # Linear layers to reduce dimensionality before bilinear fusion
        self.image_fc = nn.Linear(image_feat_dim, hidden_dim)
        self.video_fc = nn.Linear(video_feat_dim, hidden_dim)

        # Fusion layer (bilinear interaction between image and video features)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image_features, video_features):
        # Reduce the dimensionality of each feature
        image_features = self.image_fc(image_features)
        video_features = self.video_fc(video_features)
        # Compute the outer product (bilinear interaction)
        combined_features = torch.bmm(image_features.unsqueeze(2), video_features.unsqueeze(1)).view(image_features.size(0), -1)
        # Pass through fully connected layers for classification
        out = self.fc(combined_features)
        return out


class TransformerFusion(nn.Module):
    def __init__(self, image_feat_dim=2048, video_feat_dim=768, hidden_dim=768, num_heads=2, num_classes=2):
        super(TransformerFusion, self).__init__()
        # Linear layers to reduce dimensionality
        self.image_fc = nn.Linear(image_feat_dim, hidden_dim)
        self.video_fc = nn.Linear(video_feat_dim, hidden_dim)
        # Transformer encoder for feature fusion
        self.transformer = nn.Transformer(hidden_dim, num_heads)
        # Final classification layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image_features, video_features):
        # Reduce dimensionality
        image_features = self.image_fc(image_features).unsqueeze(1)  # [Batch, 1, hidden_dim]
        video_features = self.video_fc(video_features).unsqueeze(1)  # [Batch, 1, hidden_dim]
        # Concatenate image and video features into sequence for transformer
        combined_features = torch.cat((image_features, video_features), dim=1)  # [Batch, 2, hidden_dim]

        # Pass through transformer
        fused_features = self.transformer(combined_features, combined_features)
        # print("fused_features",fused_features.size())
        fused_features = fused_features.reshape([fused_features.size(0), -1])
        # Average the output of the transformer and classify
        # fused_features = fused_features.mean(dim=1)  # [Batch, hidden_dim]
        # out = self.fc(fused_features)
        return fused_features


class AttentionFusion(nn.Module):
    def __init__(self, image_feat_dim, video_feat_dim, hidden_dim):
        super(AttentionFusion, self).__init__()
        # Self-attention mechanism to compute attention weights
        self.query_image = nn.Linear(image_feat_dim, hidden_dim)
        self.key_image = nn.Linear(image_feat_dim, hidden_dim)
        self.value_image = nn.Linear(image_feat_dim, hidden_dim)
        
        self.query_video = nn.Linear(video_feat_dim, hidden_dim)
        self.key_video = nn.Linear(video_feat_dim, hidden_dim)
        self.value_video = nn.Linear(video_feat_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, image_features, video_features):
        # Compute image query, key, and value
        q_image = self.query_image(image_features)  # [Batch, hidden_dim]
        k_image = self.key_image(image_features)    # [Batch, hidden_dim]
        v_image = self.value_image(image_features)  # [Batch, hidden_dim]
        
        # Compute video query, key, and value
        q_video = self.query_video(video_features)  # [Batch, hidden_dim]
        k_video = self.key_video(video_features)    # [Batch, hidden_dim]
        v_video = self.value_video(video_features)  # [Batch, hidden_dim]
        
        # Cross-attention for image and video
        attention_weights_image = self.softmax(torch.matmul(q_image, k_video.T))  # [Batch, Batch]
        attention_weights_video = self.softmax(torch.matmul(q_video, k_image.T))  # [Batch, Batch]
        
        # Weighted sum of values
        attended_image = torch.matmul(attention_weights_image, v_video)  # [Batch, hidden_dim]
        attended_video = torch.matmul(attention_weights_video, v_image)  # [Batch, hidden_dim]
        
        # Concatenate attended features
        combined_features = torch.cat((attended_image, attended_video), dim=1)  # [Batch, 2*hidden_dim]
        return combined_features
    

class ImageStream(nn.Module):
    def __init__(self, image_model="resnet101", pretrained=True):
        super(ImageStream, self).__init__()
        # Use a pretrained ResNet for the image stream
        # self.image_model = models.resnet50(pretrained=pretrained)
        if image_model == "resnet50":
            self.image_model = models.resnet50(pretrained=pretrained)
        elif image_model == "resnet34":
            self.image_model = models.resnet34(pretrained=pretrained)
        elif image_model == "resnet101":
            self.image_model = models.resnet101(pretrained=pretrained)
            self.image_model.fc = nn.Identity()  # Remove final classification layer            
            self.out_feature = 2048
        elif image_model == "SwinB":
            self.image_model = models.swin_b(pretrained=pretrained)
            self.image_model.head = nn.Identity()  # Remove final classification layer
            self.out_feature = 1024
        self.projector = nn.Linear(self.out_feature, 512)
    def forward(self, x):
        x = self.image_model(x)  # Output shape: [Batch, 2048]
        return self.projector(x)  # Output shape: [Batch, 512]


class ImageStreamDec(nn.Module):
    def __init__(self,image_model="resnet50", pretrained=True):
        super(ImageStreamDec, self).__init__()
        # Use a pretrained ResNet for the image stream
        if image_model == "resnet50":
            self.image_model = models.resnet50(pretrained=pretrained)
            self.image_model.fc = nn.Linear(2048, 2)  # Remove final classification layer
        if image_model == "resnet34":
            self.image_model = models.resnet34(pretrained=pretrained)
            self.image_model.fc = nn.Linear(2048, 2)  # Remove final classification layer
        if image_model == "resnet101":            
            self.image_model = models.resnet101(pretrained=pretrained)
            self.image_model.fc = nn.Linear(2048, 2)  # Remove final classification layer


    def forward(self, x):
        return self.image_model(x)  # Output shape: [Batch, 2048]    

class VideoStream(nn.Module):
    def __init__(self, video_model,pretrained=True):
        super(VideoStream, self).__init__()
        # Use a pretrained ResNet3D (r3d_18 is ResNet-18 3D variant)
        if video_model == "r3d":
            self.video_model = r3d_18(pretrained=pretrained)
            self.video_model = models.video.r2plus1d_18(weights=None)
            self.video_model.stem[0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
            self.video_model.fc = nn.Identity()  # Remove final classification layer

        if video_model == "mvit":
            self.video_model = models.video.mvit_v2_s(weights='DEFAULT')
            # self.video_model.conv_proj = nn.Conv3d(1, 96, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
            self.video_model.conv_proj = nn.Conv3d(1, 96, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3))
            layer_list = list(self.video_model.children())[:-2]
            self.video_model.head = nn.Identity()  # Remove final classification layer

        # print('self.video_model',self.video_model)
        # for name, module in self.video_model._modules.items():        
        #     print("layer",name, module)

    def forward(self, x):
        vector = self.video_model(x)
        # print("VideoStream , vector.size()",vector.size())
        return  vector # Output shape: [Batch, 512]


class VideoStreamDec(nn.Module):
    def __init__(self, video_model= "r3d", pretrained=True):
        super(VideoStreamDec, self).__init__()
        # Use a pretrained ResNet3D (r3d_18 is ResNet-18 3D variant)
        if video_model == "r3d":
            self.video_model = r3d_18(pretrained=pretrained)
            self.video_model = models.video.r2plus1d_18(weights=None)
            self.video_model.stem[0] = nn.Conv3d(1, 45, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
            self.video_model.fc = nn.Linear(512, 2)  # Remove final classification layer

        if video_model == "mvit":
            self.video_model = models.video.mvit_v2_s(weights='DEFAULT')
            # self.video_model.conv_proj = nn.Conv3d(1, 96, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
            self.video_model.conv_proj = nn.Conv3d(1, 96, kernel_size=(3, 7, 7), stride=(2, 4, 4), padding=(1, 3, 3))
            layer_list = list(self.video_model.children())[:-2]
            # self.video_model.head = nn.Identity()  # Remove final classification layer
            self.video_model.head = nn.Linear(768, 2)  # Remove final classification layer


    def forward(self, x):
        vector = self.video_model(x)
        # print("vector.size()",vector.size())
        return  vector # Output shape: [Batch, 512]


class CombinedModel(nn.Module):
    def __init__(self, hidden_dim=512, num_classes=10,fusion_type="hybird",image_model = "resnet101", fusion_module = 'att',video_model="r3d",video_feat=512):
        super(CombinedModel, self).__init__()
        # Image and Video streams
        self.image_stream = ImageStream(image_model=image_model, pretrained=True)
        self.video_stream = VideoStream(pretrained=True,video_model=video_model)

        self.fusion_type = fusion_type
        self.fusion_module = fusion_module
        self.video_feat = video_feat
        # Attention fusion layer
        if "att" in self.fusion_module :
            self.attention_fusion = AttentionFusion(image_feat_dim=512, video_feat_dim=self.video_feat, hidden_dim=hidden_dim)
            self.fc = nn.Sequential(
                nn.Linear(1 * hidden_dim, 512),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(512, num_classes) )

        if "transformer" in self.fusion_module:
            self.attention_fusion = TransformerFusion(image_feat_dim=512, video_feat_dim=self.video_feat, hidden_dim=hidden_dim)
            self.fc = nn.Sequential(
                nn.Linear(1 * hidden_dim, 512),
                nn.ReLU(),
                # nn.Dropout(0.5),
                nn.Linear(512, num_classes) )


        if "hybirdd" in self.fusion_type:
            self.fc = nn.Sequential(
                nn.Linear(512+4, 256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
            )
            self.fc_image = nn.Linear(512, num_classes)
            self.fc_video = nn.Linear(self.video_feat, num_classes)
            self.fc_fusioncls = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
            )            
            return

        if "hybird" in self.fusion_type:
            self.fc = nn.Sequential(
                nn.Linear(num_classes * 3, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )
            self.fc_image = nn.Linear(512, num_classes)
            self.fc_video = nn.Linear(self.video_feat, num_classes)
            self.fc_fusioncls = nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, num_classes)
            )            
            return            

    def forward(self, image, video):
        # Forward through each stream
        image_features = self.image_stream(image)  # [Batch, 512]
        video_features = self.video_stream(video)  # [Batch, 512]
        # print("image_features",image_features.size(),"video_features.size()",video_features.size())
        if 'hybird' == self.fusion_type or 'hybirdd' == self.fusion_type:
            # Early Fusion of features and Late Fusion of logits
            fused_features = self.attention_fusion(image_features, video_features)
            # print("fused_features",fused_features.size())
            fused_prob = self.fc_fusioncls(fused_features)
            image_logits = self.fc_image(image_features)
            video_logits = self.fc_video(video_features)
            combined_logits = torch.cat((image_logits, fused_prob, video_logits), dim=1)
            # print(image_logits.size(),video_logits.size(),fused_prob.size(), combined_logits.size())
            out = self.fc(combined_logits)
            return out
        # Apply attention fusion
        combined_features = self.attention_fusion(image_features, video_features)
        # print("image_features",image_features.size(),"video_features.size()",video_features.size(),\
        #     "combined_features",combined_features.size())
        out = self.fc(combined_features)
        return out


class DecCombinedModel(nn.Module):
    def __init__(self, num_classes=2, fusion_type='average', weight_image=0.5, weight_video=0.5,video_model="r3d",image_model="resnet101"):
        super(DecCombinedModel, self).__init__()
        self.image_stream = ImageStreamDec(pretrained=True,image_model=image_model)
        self.video_stream = VideoStreamDec(pretrained=True,video_model=video_model)    
        # Fusion type can be 'average', 'weighted', or 'trainable'
        self.fusion_type = fusion_type
        self.weight_image = weight_image
        self.weight_video = weight_video

        # If fusion type is 'trainable', define a fully connected layer
        if fusion_type == 'trainable':
            self.fc_fusion = nn.Sequential(
                nn.Linear(num_classes * 2, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes) )

    def forward(self, image_input, video_input):
        # Forward pass through individual models
        # print("image_input",image_input.size(),video_input.size())
        image_output = self.image_stream(image_input)  # Shape: [Batch, num_classes]
        video_output = self.video_stream(video_input)  # Shape: [Batch, num_classes]

        if self.fusion_type == 'average':
            # Average the softmax probabilities
            # print("image_output",image_output.size(),video_output.size())
            final_output = (torch.softmax(image_output, dim=1) + torch.softmax(video_output, dim=1)) / 2

        elif self.fusion_type == 'weighted':
            # Weighted average of softmax probabilities
            final_output = (self.weight_image * torch.softmax(image_output, dim=1) +
                            self.weight_video * torch.softmax(video_output, dim=1))

        elif self.fusion_type == 'trainable':
            # Concatenate logits and pass through fusion FC layer
            combined_features = torch.cat((image_output, video_output), dim=1)
            final_output = self.fc_fusion(combined_features)

        else:
            raise ValueError("Fusion type not supported. Choose 'average', 'weighted', or 'trainable'.")
        return final_output


# Example usage
if __name__ == "__main__":
    # model = CombinedModel()
    model = DecCombinedModel()    
    # Dummy data (replace with actual images and video tensors)
    image = torch.randn(8, 1, 224, 224)  # 8 images of size 3x224x224
    video = torch.randn(8, 1, 16, 224, 224)  # 8 videos with 16 frames each, size 3x112x112

    # Forward pass
    output = model(image, video)
    print(output.shape)  # Output shape: [8, 10] for batch size 8 and 10 classes