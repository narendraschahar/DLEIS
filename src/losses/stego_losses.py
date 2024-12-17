# src/losses/stego_losses.py

class StegoLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lambda_image = config['lambda_image']
        self.lambda_message = config['lambda_message']
        
    def forward(self, cover_image, stego_image, secret_message, recovered_message):
        # Image similarity loss
        image_loss = F.mse_loss(cover_image, stego_image)
        
        # Message recovery loss
        message_loss = F.mse_loss(secret_message, recovered_message)
        
        # Combined loss
        total_loss = (
            self.lambda_image * image_loss + 
            self.lambda_message * message_loss
        )
        
        return {
            'total': total_loss,
            'image': image_loss,
            'message': message_loss
        }