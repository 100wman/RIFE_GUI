# coding:utf8
import os
import pickle

from Utils.LicenseModule import RetailValidation

from Utils.utils import Tools


class SVFIRegister(RetailValidation):
    def __init__(self):
        self.logger = Tools.get_logger("Licenser", "")
        super().__init__(self.logger)
        # TODO change pem before actual retail release
        """
        self.private_pem = b"" \
                           b"-----BEGIN RSA PRIVATE KEY-----\n" \
                           b"MIICWwIBAAKBgQCWWXMIp0clTrB4m9Lt64+Yv6MDxZuS+cRw/IhDFM87ueYcbTqZ\n" \
                           b"U1iOyWd5sk3BDbS5CsVQ45omm3bWWw1/fs7G6iafWXwEH4jCqmNjkZOmPXvswY0U\n" \
                           b"G750m+1uko35vuWj4V0tN0OIrp9A7ONPzrVi/yQtoVtruHoZHrqDF4ASGwIDAQAB\n" \
                           b"AoGABo3ltuXb8yNoDAn2+wo+21DXYW2254Rd7PMFWa9JjXgAMRMN7+szPB5JlYOR\n" \
                           b"Yi4fx8VRbsJNUQuL9bJId1tm1jH4XHawJh5SbGIv344UCDYwz4bPOAscagM9j5oA\n" \
                           b"nFqt3GkOzTVTrOwqzC6fNoqaTTRXyM8BgjbiOGiCG+9pXIkCQQC+4w+7oIjlybgh\n" \
                           b"6QIYGQt3zbsT56K8ae84EqsKTGm4u7KVkbCjPRx/SneM0TJhSSSVbjCbvRl25C+3\n" \
                           b"3hTCVpQNAkEAyaKAvOUtDFubR33mAP92SIBAljFUIbsbaz2Fp5lA4Jmr3CDyCcaa\n" \
                           b"E5Qx/udy1kYt3jKdV9jQNHbh5jt2K9PsxwJATMfWVzked5do+jLYRcslIr5c5ofA\n" \
                           b"nJrbvyk7JTxRNh5BmgntC+wT31ubtMecxSb/kR+ua6ZnbLwiOYoZvYXHrQJAEQql\n" \
                           b"/NEVzJyVdCZk4SK2OYx1aPxEUxGAUMEDYdXnENSMHO+/5Sme7haxXwzqvMdzqvr2\n" \
                           b"J22Qs05060ONSkkAEwJADGBeXhf5cwxtktbZGC1+TvtQJwlcTDLIjecziDhCD98i\n" \
                           b"/Z88zsJxYoxy0ZZSIItEw+S2GtWGVj6TIQNmZlLZ/A==\n" \
                           b"-----END RSA PRIVATE KEY-----"
        self.public_pem = b""
        """
        self._rsa_worker.public_pem = \
            b"-----BEGIN PUBLIC KEY-----\n" \
            b"MIGfMA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCWWXMIp0clTrB4m9Lt64+Yv6MD\n" \
            b"xZuS+cRw/IhDFM87ueYcbTqZU1iOyWd5sk3BDbS5CsVQ45omm3bWWw1/fs7G6iaf\n" \
            b"WXwEH4jCqmNjkZOmPXvswY0UG750m+1uko35vuWj4V0tN0OIrp9A7ONPzrVi/yQt\n" \
            b"oVtruHoZHrqDF4ASGwIDAQAB\n" \
            b"-----END PUBLIC KEY-----"

    def ValidateRegisterBin(self, bin_path):
        bin_data = pickle.load(open(bin_path, 'rb'))
        assert type(bin_data) is dict, "Type of License Data is not correct"
        license_data = self._rsa_worker.decrypt_with_private_key(bin_data.get('license_data'))
        bin_data = {'license_key': self._rsa_worker.encrypt_with_private_key(license_data)}
        pickle.dump(bin_data, open(bin_path, 'wb'))
        print(f"License: {license_data} to {bin_data}")
        return True


if __name__ == '__main__':
    print("""
          _____                    _____                    _____                    _____                                    _____            _____                    _____                    _____                    _____                    _____                    _____                    _____          
         /\    \                  /\    \                  /\    \                  /\    \                                  /\    \          /\    \                  /\    \                  /\    \                  /\    \                  /\    \                  /\    \                  /\    \         
        /::\    \                /::\____\                /::\    \                /::\    \                                /::\____\        /::\    \                /::\    \                /::\    \                /::\____\                /::\    \                /::\    \                /::\    \        
       /::::\    \              /:::/    /               /::::\    \               \:::\    \                              /:::/    /        \:::\    \              /::::\    \              /::::\    \              /::::|   |               /::::\    \              /::::\    \              /::::\    \       
      /::::::\    \            /:::/    /               /::::::\    \               \:::\    \                            /:::/    /          \:::\    \            /::::::\    \            /::::::\    \            /:::::|   |              /::::::\    \            /::::::\    \            /::::::\    \      
     /:::/\:::\    \          /:::/    /               /:::/\:::\    \               \:::\    \                          /:::/    /            \:::\    \          /:::/\:::\    \          /:::/\:::\    \          /::::::|   |             /:::/\:::\    \          /:::/\:::\    \          /:::/\:::\    \     
    /:::/__\:::\    \        /:::/____/               /:::/__\:::\    \               \:::\    \                        /:::/    /              \:::\    \        /:::/  \:::\    \        /:::/__\:::\    \        /:::/|::|   |            /:::/__\:::\    \        /:::/__\:::\    \        /:::/__\:::\    \    
    \:::\   \:::\    \       |::|    |               /::::\   \:::\    \              /::::\    \                      /:::/    /               /::::\    \      /:::/    \:::\    \      /::::\   \:::\    \      /:::/ |::|   |            \:::\   \:::\    \      /::::\   \:::\    \      /::::\   \:::\    \   
  ___\:::\   \:::\    \      |::|    |     _____    /::::::\   \:::\    \    ____    /::::::\    \                    /:::/    /       ____    /::::::\    \    /:::/    / \:::\    \    /::::::\   \:::\    \    /:::/  |::|   | _____    ___\:::\   \:::\    \    /::::::\   \:::\    \    /::::::\   \:::\    \  
 /\   \:::\   \:::\    \     |::|    |    /\    \  /:::/\:::\   \:::\    \  /\   \  /:::/\:::\    \                  /:::/    /       /\   \  /:::/\:::\    \  /:::/    /   \:::\    \  /:::/\:::\   \:::\    \  /:::/   |::|   |/\    \  /\   \:::\   \:::\    \  /:::/\:::\   \:::\    \  /:::/\:::\   \:::\____\ 
/::\   \:::\   \:::\____\    |::|    |   /::\____\/:::/  \:::\   \:::\____\/::\   \/:::/  \:::\____\                /:::/____/       /::\   \/:::/  \:::\____\/:::/____/     \:::\____\/:::/__\:::\   \:::\____\/:: /    |::|   /::\____\/::\   \:::\   \:::\____\/:::/__\:::\   \:::\____\/:::/  \:::\   \:::|    |
\:::\   \:::\   \::/    /    |::|    |  /:::/    /\::/    \:::\   \::/    /\:::\  /:::/    \::/    /                \:::\    \       \:::\  /:::/    \::/    /\:::\    \      \::/    /\:::\   \:::\   \::/    /\::/    /|::|  /:::/    /\:::\   \:::\   \::/    /\:::\   \:::\   \::/    /\::/   |::::\  /:::|____|
 \:::\   \:::\   \/____/     |::|    | /:::/    /  \/____/ \:::\   \/____/  \:::\/:::/    / \/____/                  \:::\    \       \:::\/:::/    / \/____/  \:::\    \      \/____/  \:::\   \:::\   \/____/  \/____/ |::| /:::/    /  \:::\   \:::\   \/____/  \:::\   \:::\   \/____/  \/____|:::::\/:::/    / 
  \:::\   \:::\    \         |::|____|/:::/    /            \:::\    \       \::::::/    /                            \:::\    \       \::::::/    /            \:::\    \               \:::\   \:::\    \              |::|/:::/    /    \:::\   \:::\    \       \:::\   \:::\    \            |:::::::::/    /  
   \:::\   \:::\____\        |:::::::::::/    /              \:::\____\       \::::/____/                              \:::\    \       \::::/____/              \:::\    \               \:::\   \:::\____\             |::::::/    /      \:::\   \:::\____\       \:::\   \:::\____\           |::|\::::/    /   
    \:::\  /:::/    /        \::::::::::/____/                \::/    /        \:::\    \                               \:::\    \       \:::\    \               \:::\    \               \:::\   \::/    /             |:::::/    /        \:::\  /:::/    /        \:::\   \::/    /           |::| \::/____/    
     \:::\/:::/    /          ~~~~~~~~~~                       \/____/          \:::\    \                               \:::\    \       \:::\    \               \:::\    \               \:::\   \/____/              |::::/    /          \:::\/:::/    /          \:::\   \/____/            |::|  ~|          
      \::::::/    /                                                              \:::\    \                               \:::\    \       \:::\    \               \:::\    \               \:::\    \                  /:::/    /            \::::::/    /            \:::\    \                |::|   |          
       \::::/    /                                                                \:::\____\                               \:::\____\       \:::\____\               \:::\____\               \:::\____\                /:::/    /              \::::/    /              \:::\____\               \::|   |          
        \::/    /                                                                  \::/    /                                \::/    /        \::/    /                \::/    /                \::/    /                \::/    /                \::/    /                \::/    /                \:|   |          
         \/____/                                                                    \/____/                                  \/____/          \/____/                  \/____/                  \/____/                  \/____/                  \/____/                  \/____/                  \|___|          
                                                                                                                                                                                                                                                                                                                    
    """)
    print("\nVersion: 2.0.0")
    reg = SVFIRegister()
    while True:
        _bin_path = input()
        if not len(_bin_path):
            print("[INFO] Type a key to exit")
            exit(0)
        if not os.path.exists(_bin_path):
            print("[Error] Invalid bin_path")
            continue
        reg.ValidateRegisterBin(_bin_path)
