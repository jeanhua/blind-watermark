package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

var embedCmd = &cobra.Command{
	Use:   "embed [flags] <orig_image> <watermark_text> <output_image>",
	Short: "Embed a text watermark into an image",
	Long: `Embed a text watermark into an image.
Example: blind_watermark embed --pwd 1234 input.jpg "my secret" output.png`,
	Args: cobra.ExactArgs(3),
	RunE: runEmbed,
}

func init() {
	RootCmd.AddCommand(embedCmd)
	embedCmd.Flags().StringVarP(&password, "pwd", "p", "1", "Password for embedding (used to seed the shuffle)")
	embedCmd.Flags().StringVar(&passwordWM, "pwd-wm", "", "Password for watermark encryption (defaults to pwd)")
}

func runEmbed(cmd *cobra.Command, args []string) error {
	inputFile := args[0]
	wmText := args[1]
	outputFile := args[2]

	pwdIMG, pwdWM, err := parsePasswords()
	if err != nil {
		return err
	}

	img, err := ReadImageFile(inputFile)
	if err != nil {
		return fmt.Errorf("read image: %w", err)
	}

	wm := NewWaterMark(pwdWM, pwdIMG)
	wm.ReadImageFromBytes(img)
	wm.ReadWatermarkString(wmText)

	fmt.Printf("Watermark bits length: %d\n", len(wmText))
	fmt.Printf("Embedded bits count: %d\n", wm.WmSize())

	result := wm.Embed()
	if err := WriteImageFile(outputFile, result); err != nil {
		return fmt.Errorf("write image: %w", err)
	}
	fmt.Println("Embed succeeded! Output:", outputFile)
	return nil
}
