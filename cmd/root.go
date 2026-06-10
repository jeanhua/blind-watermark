package cmd

import (
	"blind_watermark_go/internal/bwm"
	"blind_watermark_go/internal/imageutil"
	"fmt"
	"strconv"

	"github.com/spf13/cobra"
)

var (
	password   string
	wmShape    string
	passwordWM string
)

var RootCmd = &cobra.Command{
	Use:   "blind_watermark",
	Short: "Blind watermark tool - embed or extract invisible watermarks in images",
	Long: `A Go implementation of the blind watermark algorithm.
Embeds text watermarks into images using DWT + DCT + SVD.
Extracts watermarks from watermarked images.`,
	Run: func(cmd *cobra.Command, args []string) {
		cmd.Help()
	},
}

func Execute() {
	if err := RootCmd.Execute(); err != nil {
		fmt.Fprintln(RootCmd.ErrOrStderr(), err)
	}
}

func parsePasswords() (imgPwd, wmPwd uint64, err error) {
	imgPwd, err = strconv.ParseUint(password, 10, 64)
	if err != nil {
		return 0, 0, fmt.Errorf("invalid password: %w", err)
	}
	wmPwd = imgPwd
	if passwordWM != "" {
		wmPwd, err = strconv.ParseUint(passwordWM, 10, 64)
		if err != nil {
			return 0, 0, fmt.Errorf("invalid watermark password: %w", err)
		}
	}
	return
}

// Helpers exposed for subcommands
func ReadImageFile(path string) ([][][3]float32, error) {
	return imageutil.ReadImage(path)
}

func WriteImageFile(path string, data [][][3]float32) error {
	return imageutil.WriteImage(path, data)
}

func NewWaterMark(pwdWM, pwdIMG uint64) *bwm.WaterMark {
	return bwm.NewWaterMark(pwdWM, pwdIMG)
}
