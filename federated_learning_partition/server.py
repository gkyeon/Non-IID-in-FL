import flwr as fl
import numpy as np

# SaveModelStrategy 클래스 정의 (모델 파라미터 저장)
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        # 클라이언트 업데이트의 집계를 수행
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        
        # 집계된 파라미터가 있는지 확인
        if aggregated_parameters is not None:
            print(f"Saving round {rnd} aggregated parameters and metrics...")

            # 집계된 파라미터를 numpy ndarray 리스트로 변환
            parameter_ndarrays = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # 파라미터를 .npz 파일로 저장
            np.savez(f"round-{rnd}-parameters.npz", *parameter_ndarrays)

            # 메트릭을 .npz 파일로 저장
            np.savez(f"round-{rnd}-metrics.npz", **aggregated_metrics)

        return aggregated_parameters, aggregated_metrics

# 연합학습 서버 실행
strategy = SaveModelStrategy(
    fraction_fit=1,  # 다음 라운드에 100% 클라이언트만 샘플링
    min_fit_clients=2,  # 최소 2개의 클라이언트가 샘플링되어야 함
    min_available_clients=2,  # 총 최소 2개의 클라이언트 필요
)

if __name__ == "__main__":
    # ServerConfig를 사용하여 라운드 수 설정
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=100),  # 라운드 수를 ServerConfig로 설정
        strategy=strategy
    )
