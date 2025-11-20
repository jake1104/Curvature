
import numpy as np

def kappa(T, s):
  """접선 벡터 T(s)로부터 곡률 계산"""
  dT_ds = np.gradient(T, s, axis=0)
  kappa_s = np.linalg.norm(dT_ds, axis=1)
  return kappa_s

def reverse_kappa(kappa_s, s, target_angle=np.pi/2, T0=None, X0=None):
  """
  곡률 배열로부터 곡선 복원

  Parameters
  ----------
  kappa_s : np.ndarray
    곡률 배열
  s : np.ndarray
    아크길이 배열
  target_angle : float
    최종 곡선의 꺾임 각도 (라디안)
  T0 : np.ndarray
    초기 접선 벡터, 기본 [1,0]
  X0 : np.ndarray
    초기 위치, 기본 [0,0]

  Returns
  -------
  X : np.ndarray
    복원된 곡선 좌표 (Nx2)
  T : np.ndarray
    접선 벡터 (Nx2)
  theta : np.ndarray
    접선 각도 (Nx,)
  """
  if T0 is None:
    T0 = np.array([1.0, 0.0])
  if X0 is None:
    X0 = np.array([0.0, 0.0])

  # 전체 꺾임 스케일링
  total_angle = np.sum(kappa_s * np.gradient(s))
  scale = target_angle / total_angle if total_angle != 0 else 1.0
  kappa_s_scaled = kappa_s * scale

  # 1) 곡률 적분 → θ(s)
  theta = np.cumsum(kappa_s_scaled * np.gradient(s))
  theta -= theta[0]  # 초기 각도 0으로 맞춤

  # 2) 접선 벡터 T(s) 생성
  T = np.column_stack((np.cos(theta), np.sin(theta)))

  # 초기 방향 적용 (회전)
  R = np.array([[T0[0], -T0[1]],
                [T0[1],  T0[0]]])
  T = T @ R.T

  # 3) 곡선 좌표 적분
  d_s = np.gradient(s)
  X = np.zeros((len(s), 2))
  X[0] = X0

  for i in range(1, len(s)):
    X[i] = X[i-1] + T[i] * d_s[i]

  return X, T, theta

def curvature_to_curve(kappa_s, s=None, target_angle=np.pi/2):
  """
  곡률 배열 입력 → 자동 스케일링 → 2D 곡선 반환

  Parameters
  ----------
  kappa_s : np.ndarray
    곡률 배열
  s : np.ndarray, optional
    아크길이 배열, None이면 0~1로 생성
  target_angle : float
    목표 꺾임 각도 (라디안)

  Returns
  -------
  X : np.ndarray
    곡선 좌표 (Nx2)
  T : np.ndarray
    접선 벡터 (Nx2)
  theta : np.ndarray
    접선 각도
  """
  if s is None:
    s = np.linspace(0, 1, len(kappa_s))
  return reverse_kappa(kappa_s, s, target_angle=target_angle)


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  s = np.linspace(0, 10, 200)

  curvature_raw = np.ones_like(s)
  X, T, theta = curvature_to_curve(curvature_raw, s, target_angle=np.pi/2)
  plt.subplot(321)
  plt.plot(X[:,0], X[:,1])
  plt.axis('equal')
  plt.title('Circle')

  curvature_raw = np.sin(np.pi * s / 10)
  X, T, theta = curvature_to_curve(curvature_raw, s, target_angle=np.pi/2)
  plt.subplot(322)
  plt.plot(X[:,0], X[:,1])
  plt.axis('equal')
  plt.title('Sine Wave')

  from NormalDistribution import NormalDistribution
  nd = NormalDistribution(mean=5)
  curvature_raw = nd.N(s)
  X, T, theta = curvature_to_curve(curvature_raw, s, target_angle=np.pi/2)
  plt.subplot(323)
  plt.plot(X[:,0], X[:,1])
  plt.axis('equal')
  plt.title('Normal Distribution')

  sigmoid = lambda x: 1 / (1 + np.exp(-x))
  sigmoid_deriv = lambda x: sigmoid(x) * (1 - sigmoid(x))
  curvature_raw = sigmoid_deriv(s - 5)
  X, T, theta = curvature_to_curve(curvature_raw, s, target_angle=np.pi/2)
  plt.subplot(324)
  plt.plot(X[:,0], X[:,1])
  plt.axis('equal')
  plt.title('Sigmoid Derivative')
  
  tanh = lambda x: np.tanh(x)
  tanh_deriv = lambda x: 1 - tanh(x)**2
  curvature_raw = tanh_deriv(s - 5)
  X, T, theta = curvature_to_curve(curvature_raw, s, target_angle=np.pi/2)
  plt.subplot(325)
  plt.plot(X[:,0], X[:,1])
  plt.axis('equal')
  plt.title('Tanh Derivative')

  plt.savefig('example.png')

  plt.show()
