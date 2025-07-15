async def forward_179(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1.1: Extract constants (SC_CoT)
    sc_instruction = "Sub-task 1.1: Extract all physical parameters and explicit constants: number of charges = 13, q = 2e, e = 1.602e-19 C, k = 8.988e9 N·m²/C², R = 2 m, E0(12) = 23.33."  
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {"subtask_id":"subtask_1.1","instruction":sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents:
        thinking, answer = await agent([taskInfo], sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_decision_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_agent1([taskInfo] + possible_thinkings + possible_answers, "Sub-task 1.1: Synthesize and choose the most consistent constants.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc1['response'] = {"thinking":thinking1_1,"answer":answer1_1}
    logs.append(subtask_desc1)
    print("Step 1.1:", sub_tasks[-1])

    # Stage 1.2: Compute k·e² (SC_CoT)
    sc2_instruction = "Sub-task 1.2: Compute k·e² with k=8.988e9 and e=1.602e-19; perform reflexion check of ~2.307e-28 J·m."  
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id":"subtask_1.2","instruction":sc2_instruction,"context":["user query","thinking1_1","answer1_1"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents:
        thinking, answer = await agent([taskInfo, thinking1_1, answer1_1], sc2_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings2.append(thinking)
        possible_answers2.append(answer)
    final_decision_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_2, answer1_2 = await final_decision_agent2([taskInfo] + possible_thinkings2 + possible_answers2, "Sub-task 1.2: Synthesize and choose the most consistent k·e² value.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1.2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc2['response'] = {"thinking":thinking1_2,"answer":answer1_2}
    logs.append(subtask_desc2)
    print("Step 1.2:", sub_tasks[-1])

    # Stage 2.1: Center–peripheral energy U_cp (Reflexion)
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 2.1: Compute U_cp = 12·k·(2e)²/R using k·e² from Sub-task 1.2 and constants." + reflect_inst
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs = [taskInfo, thinking1_2, answer1_2]
    subtask_desc3 = {"subtask_id":"subtask_2.1","instruction":cot_reflect_instruction,"context":["user query","thinking1_2","answer1_2"],"agent_collaboration":"Reflexion"}
    thinking2_1, answer2_1 = await cot_agent(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking2_1, answer2_1], "Please review and provide limitations. If correct output 'True'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking2_1, answer2_1, feedback])
        thinking2_1, answer2_1 = await cot_agent(cot_inputs, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
    sub_tasks.append(f"Sub-task 2.1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc3['response'] = {"thinking":thinking2_1,"answer":answer2_1}
    logs.append(subtask_desc3)
    print("Step 2.1:", sub_tasks[-1])

    # Stage 3.1: Peripheral–peripheral energy U_th (Debate)
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction = "Sub-task 3.1: Compute U_th = E0(12)·k·(2e)²/R using E0 from Sub-task 1.1 and k·e² from Sub-task 1.2." + debate_instr
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking = [[] for _ in range(self.max_round)]
    all_answer = [[] for _ in range(self.max_round)]
    subtask_desc4 = {"subtask_id":"subtask_3.1","instruction":debate_instruction,"context":["user query","thinking2_1","answer2_1"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for agent in debate_agents:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking2_1, answer2_1], debate_instruction, r, is_sub_task=True)
            else:
                thinking, answer = await agent([taskInfo, thinking2_1, answer2_1] + all_thinking[r-1] + all_answer[r-1], debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3_1, answer3_1 = await final_decision_agent3([taskInfo, thinking2_1, answer2_1] + all_thinking[-1] + all_answer[-1], "Sub-task 3.1: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent3.id}, thinking: {thinking3_1.content}; answer: {answer3_1.content}")
    sub_tasks.append(f"Sub-task 3.1 output: thinking - {thinking3_1.content}; answer - {answer3_1.content}")
    subtask_desc4['response'] = {"thinking":thinking3_1,"answer":answer3_1}
    logs.append(subtask_desc4)
    print("Step 3.1:", sub_tasks[-1])

    # Stage 4.1: Sum energies U_total (Reflexion)
    reflect_inst2 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction2 = "Sub-task 4.1: Sum U_cp and U_th to compute U_total." + reflect_inst2
    cot_agent2 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent2 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs2 = [taskInfo, thinking2_1, answer2_1, thinking3_1, answer3_1]
    subtask_desc5 = {"subtask_id":"subtask_4.1","instruction":cot_reflect_instruction2,"context":["user query","thinking2_1","answer2_1","thinking3_1","answer3_1"],"agent_collaboration":"Reflexion"}
    thinking4_1, answer4_1 = await cot_agent2(cot_inputs2, cot_reflect_instruction2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent2.id}, thinking: {thinking4_1.content}; answer: {answer4_1.content}")
    for i in range(self.max_round):
        feedback, correct = await critic_agent2([taskInfo, thinking4_1, answer4_1], "Please review and provide limitations. If correct output 'True'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent2.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs2.extend([thinking4_1, answer4_1, feedback])
        thinking4_1, answer4_1 = await cot_agent2(cot_inputs2, cot_reflect_instruction2, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent2.id}, thinking: {thinking4_1.content}; answer: {answer4_1.content}")
    sub_tasks.append(f"Sub-task 4.1 output: thinking - {thinking4_1.content}; answer - {answer4_1.content}")
    subtask_desc5['response'] = {"thinking":thinking4_1,"answer":answer4_1}
    logs.append(subtask_desc5)
    print("Step 4.1:", sub_tasks[-1])

    # Stage 4.2: Select closest choice (SC_CoT)
    sc4_instruction = "Sub-task 4.2: Compare U_total to the provided choices and select the closest match, justifying differences explicitly to three decimals."  
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc6 = {"subtask_id":"subtask_4.2","instruction":sc4_instruction,"context":["user query","thinking4_1","answer4_1"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents:
        thinking, answer = await agent([taskInfo, thinking4_1, answer4_1], sc4_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings4.append(thinking)
        possible_answers4.append(answer)
    final_decision_agent4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4_2, answer4_2 = await final_decision_agent4([taskInfo] + possible_thinkings4 + possible_answers4, "Sub-task 4.2: Synthesize and choose the most consistent final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4.2 output: thinking - {thinking4_2.content}; answer - {answer4_2.content}")
    subtask_desc6['response'] = {"thinking":thinking4_2,"answer":answer4_2}
    logs.append(subtask_desc6)
    print("Step 4.2:", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4_2, answer4_2, sub_tasks, agents)
    return final_answer, logs