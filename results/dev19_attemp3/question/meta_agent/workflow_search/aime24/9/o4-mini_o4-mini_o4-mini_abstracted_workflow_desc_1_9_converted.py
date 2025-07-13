async def forward_9(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Sub-task 1: Chain-of-Thought
    cot_instruction = "Sub-task 1: Formulate the probability formula for P(K=k) using hypergeometric distribution for k from 0 to 4."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_instruction,"context":["user query"],"agent_collaboration":"CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, formulating hypergeometric formula, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response']={"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Sub-task 2: Self-Consistency Chain-of-Thought
    cot_sc_instruction = "Sub-task 2: Define P(prize win)=P(K>=2) and P(grand prize)=P(K=4), and express conditional probability P(K=4)/P(K>=2)."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction,"context":["user query","thinking of subtask_1","answer of subtask_1"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        t2,a2 = await cot_agents[i]([taskInfo,thinking1,answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, defining conditional probability formula, thinking: {t2.content}; answer: {a2.content}")
        possible_thinkings.append(t2)
        possible_answers.append(a2)
    final_decision_agent_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo,thinking1,answer1]+possible_thinkings+possible_answers,
                                                    "Sub-task 2: Synthesize and choose the most consistent conditional probability expression.",
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response']={"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Sub-task 3: Chain-of-Thought
    cot_instruction3 = "Sub-task 3: Calculate P(K=2), P(K=3), and P(K=4) numerically using combinations for the lottery scenario."
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_instruction3,"context":["user query","thinking of subtask_1","answer of subtask_1"],"agent_collaboration":"CoT"}
    thinking3, answer3 = await cot_agent3([taskInfo,thinking1,answer1], cot_instruction3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent3.id}, computing numerical probabilities, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response']={"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    # Sub-task 4: Self-Consistency Chain-of-Thought
    cot_sc_instruction4 = "Sub-task 4: Compute P(K>=2)=P(K=2)+P(K=3)+P(K=4), form ratio P(K=4)/P(K>=2), and simplify to lowest terms."
    N = self.max_sc
    cot_agents4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_sc_instruction4,"context":["user query","thinking of subtask_2","answer of subtask_2","thinking of subtask_3","answer of subtask_3"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        t4,a4 = await cot_agents4[i]([taskInfo,thinking2,answer2,thinking3,answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, simplifying fraction, thinking: {t4.content}; answer: {a4.content}")
        possible_thinkings4.append(t4)
        possible_answers4.append(a4)
    final_decision_agent4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4([taskInfo,thinking2,answer2,thinking3,answer3]+possible_thinkings4+possible_answers4,
                                                    "Sub-task 4: Synthesize and choose the most consistent simplified fraction and compute m+n.",
                                                    is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response']={"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    # Sub-task 5: Chain-of-Thought
    cot_instruction5 = "Sub-task 5: Provide the final result by calculating m+n from the simplified fraction obtained in Sub-task 4."
    cot_agent5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":cot_instruction5,"context":["user query","thinking of subtask_4","answer of subtask_4"],"agent_collaboration":"CoT"}
    thinking5, answer5 = await cot_agent5([taskInfo,thinking4,answer4], cot_instruction5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent5.id}, computing m+n, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response']={"thinking":thinking5,"answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs