import React from "react";
import { Card } from "antd";
import { ArrowRightOutlined, AuditOutlined, LockOutlined, ScanOutlined, UserOutlined } from "@ant-design/icons";
import "./portal.css";

const entrances = [
  { id: "consumer", title: "我是消费者", detail: "核对产品宣传与功效依据", icon: UserOutlined },
  { id: "reviewer", title: "我是企业审核人员", detail: "审核宣传合规与退款破损线索", icon: AuditOutlined },
];

export default function Portal() {
  return <div className="portal">
    <header className="portal-top portal-width">
      <a className="portal-skip" href="#choose-role">跳到身份选择</a>
      <div className="portal-brand"><ScanOutlined aria-hidden="true" /> BeautyProof</div>
      <span className="portal-local"><LockOutlined aria-hidden="true" /> 图片在本地处理</span>
    </header>
    <main>
      <section className="portal-hero" aria-labelledby="portal-title">
        <img className="portal-image" src="/static/beautyproof/optical-hero.webp" alt=""
          width="1536" height="1024" fetchPriority="high" />
        <div className="portal-width portal-content" data-enter>
          <h1 id="portal-title"><span>美妆判断，</span><span>有据可循。</span></h1>
          <p className="portal-lead">发来一段宣传文案或一张图片。<br className="portal-break" />让审核助手找出原句、图像和可复核的依据。</p>
          <nav id="choose-role" className="portal-roles" aria-label="选择使用身份" tabIndex={-1}>
            {entrances.map(({ id, title, detail, icon: Icon }) => <a key={id}
              className={"portal-entry portal-entry-" + id} href={"#" + id}>
              <Card className="portal-card" variant="outlined">
                <Icon className="portal-card-icon" aria-hidden="true" />
                <div className="portal-card-body"><strong>{title}</strong><span>{detail}</span></div>
                <ArrowRightOutlined className="portal-card-arrow" aria-hidden="true" />
              </Card>
            </a>)}
          </nav>
        </div>
      </section>
    </main>
  </div>;
}
