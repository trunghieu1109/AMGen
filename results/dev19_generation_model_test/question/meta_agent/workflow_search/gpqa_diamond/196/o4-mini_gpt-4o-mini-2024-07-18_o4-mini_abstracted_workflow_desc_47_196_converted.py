async def forward_196(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    sc_instruction = "Sub-task 1: Interpret the IR spectrum data, identifying functional groups with context from user query"
    N = self.max_sc
    sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking1_i, answer1_i = await sc_agents[i]([taskInfo], sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents[i].id}, analyze IR, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr1 = "Sub-task 1: Synthesize and choose the most consistent interpretation of the IR data"
    thinking1, answer1 = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1, final_instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to deduce the ¹H NMR pattern and aliphatic substituent."
    cot_reflect_instruction = "Sub-task 2: Analyze the ¹H NMR data to deduce the aromatic substitution pattern and aliphatic substituent." + reflect_inst
    cot_agent2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_reflect_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "Reflexion"}
    cot_inputs2 = [taskInfo, thinking1, answer1]
    thinking2, answer2 = await cot_agent2(cot_inputs2, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent2.id}, initial NMR analysis, thinking: {thinking2.content}; answer: {answer2.content}")
    for i in range(self.max_round):
        feedback2, correct2 = await critic_agent2([taskInfo, thinking2, answer2], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent2.id}, feedback: {feedback2.content}; correct: {correct2.content}")
        if correct2.content == "True":
            break
        cot_inputs2.extend([thinking2, answer2, feedback2])
        thinking2, answer2 = await cot_agent2(cot_inputs2, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent2.id}, refined NMR analysis, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr1 = "Sub-task 3: Determine the reaction outcome of red phosphorus/HI on a benzoic acid derivative." + "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(self.max_round)]
    all_answer3 = [[] for _ in range(self.max_round)]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": debate_instr1, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents3:
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instr1, r, is_sub_task=True)
            else:
                input_infos3 = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(input_infos3, debate_instr1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking3, answer3 = await final_decision_agent3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], final_instr3, is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_instruction4 = "Sub-task 4: Compare the predicted hydrocarbon structure with the answer choices and select the correct final product."
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction4, "context": ["user query","thinking of subtask 3","answer of subtask 3"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, compare structures, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs