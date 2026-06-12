package cmd

import (
	"fmt"
	"strconv"

	"github.com/spf13/cobra"
)

var extractCmd = &cobra.Command{
	Use:   "extract [flags] <watermarked_image>",
	Short: "Extract a text watermark from an image",
	Long: `Extract a text watermark from an image.
Example: blind_watermark extract --pwd 1234 --wm-shape 111 output.png`,
	Args: cobra.ExactArgs(1),
	RunE: runExtract,
}

func init() {
	RootCmd.AddCommand(extractCmd)
	extractCmd.Flags().StringVarP(&password, "pwd", "p", "1", "Password for extraction (must match embed password)")
	extractCmd.Flags().StringVar(&passwordWM, "pwd-wm", "", "Password for watermark encryption (defaults to pwd)")
	extractCmd.Flags().StringVar(&wmShape, "wm-shape", "", "Watermark length in bits (required for extraction)")
	extractCmd.MarkFlagRequired("wm-shape")
}

func runExtract(cmd *cobra.Command, args []string) error {
	inputFile := args[0]

	pwdIMG, pwdWM, err := parsePasswords()
	if err != nil {
		return err
	}

	wmLen, err := strconv.Atoi(wmShape)
	if err != nil {
		return fmt.Errorf("invalid wm-shape: %w", err)
	}

	img, err := ReadImageFile(inputFile)
	if err != nil {
		return fmt.Errorf("read image: %w", err)
	}

	wm := NewWaterMark(pwdWM, pwdIMG)
	raw := wm.ExtractRaw(img)
	wmStr := wm.ExtractStringFromRaw(raw, wmLen)

	fmt.Println("Extract succeeded! Watermark is:")
	fmt.Println(wmStr)
	return nil
}
