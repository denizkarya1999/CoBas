from TouchUI import dialogs


def show_about_window(parent=None):
    return dialogs.showinfo("About CoBas", 'CoBas Battery Reader · Version 1.0\n\nCoBas is a contactless battery sensing prototype that combines near-ultrasonic acoustic signals, IWR6843AOP range-angle responses, and synchronized thermal imaging to support lithium-ion battery state-of-charge monitoring.\n\nAuthors\nDeniz Karya Acikbas\nPedro Callado de Paiva\nSelase Doku\nNitin Shankar Madhu\nChuka Ezeoke\nClark Friese\nAhmad Jayeb\nLucas Hammermeister\nNingyue Mao\nXuan Zhou\nXiao Zhang\n\n© 2026 Trustworthy AIoT Lab, University of Michigan-Dearborn.\nAll rights reserved.', parent=parent)
