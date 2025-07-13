async def forward_154(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction1 = "Sub-task 1: Extract and summarize the matrix representations of P_x, P_y, P_z and the system’s state vector." + reflect_inst
    cot_agent1 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent1 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_reflect_instruction1,"context":["user query"],"agent_collaboration":"Reflexion"}
    thinking1, answer1 = await cot_agent1([taskInfo], cot_reflect_instruction1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent1.id}, extracting and summarizing given information, thinking: {thinking1.content}; answer: {answer1.content}")
    critic_inst1 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(self.max_round):
        feedback1, correct1 = await critic_agent1([taskInfo, thinking1, answer1], critic_inst1, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent1.id}, providing feedback, thinking: {feedback1.content}; answer: {correct1.content}")
        if correct1.content == "True":
            break
        thinking1, answer1 = await cot_agent1([taskInfo, thinking1, answer1, feedback1], cot_reflect_instruction1, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent1.id}, refining extraction, thinking: {thinking1.content}; answer: {answer1.content}")
    subtask_desc1["response"] = {"thinking":thinking1,"answer":answer1}
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction2 = "Sub-task 2: Based on the output from Sub-task 1, compute the expectation value of P_z = psi^T P_z psi."
    N = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction2,"context":["user query","thinking of subtask_1","answer of subtask_1"],"agent_collaboration":"SC_CoT"}
    possible_thinkings2 = []
    possible_answers2 = []
    for i in range(N):
        thinking2, answer2 = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, computing expectation value of P_z, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings2.append(thinking2)
        possible_answers2.append(answer2)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, "Sub-task 2: Synthesize and choose the most consistent computation of the expectation value of P_z.", is_sub_task=True)
    subtask_desc2["response"] = {"thinking":thinking2,"answer":answer2}
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction3 = "Sub-task 3: Based on the output from Sub-task 1, compute the second moment ⟨P_z^2⟩ = psi^T P_z^2 psi."
    cot_agents3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":cot_sc_instruction3,"context":["user query","thinking of subtask_1","answer of subtask_1"],"agent_collaboration":"SC_CoT"}
    possible_thinkings3 = []
    possible_answers3 = []
    for i in range(N):
        thinking3, answer3 = await cot_agents3[i]([taskInfo, thinking1, answer1], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3[i].id}, computing second moment of P_z, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_thinkings3.append(thinking3)
        possible_answers3.append(answer3)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking1, answer1] + possible_thinkings3 + possible_answers3, "Sub-task 3: Synthesize and choose the most consistent computation of ⟨P_z^2⟩.", is_sub_task=True)
    subtask_desc3["response"] = {"thinking":thinking3,"answer":answer3}
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction4 = "Sub-task 4: Based on the outputs from Sub-task 2 and Sub-task 3, compute the uncertainty ΔP_z = sqrt(⟨P_z^2⟩ - ⟨P_z⟩^2) and match to provided choices."
    cot_agents4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_sc_instruction4,"context":["user query","thinking of subtask_2","answer of subtask_2","thinking of subtask_3","answer of subtask_3"],"agent_collaboration":"SC_CoT"}
    possible_thinkings4 = []
    possible_answers4 = []
    for i in range(N):
        thinking4, answer4 = await cot_agents4[i]([taskInfo, thinking2, answer2, thinking3, answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents4[i].id}, computing uncertainty of P_z, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_thinkings4.append(thinking4)
        possible_answers4.append(answer4)
    final_decision_agent4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4([taskInfo, thinking2, answer2, thinking3, answer3] + possible_thinkings4 + possible_answers4, "Sub-task 4: Synthesize and choose the most consistent computation of ΔP_z matching provided choices.", is_sub_task=True)
    subtask_desc4["response"] = {"thinking":thinking4,"answer":answer4}
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs