async def forward_190(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_1 = "Sub-task 1: Predict the structure of product 1 formed by deprotonation of the 3-hydroxymethyl group and benzylation with benzyl bromide."
    N1 = self.max_sc
    cot_sc_agents_1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings_1 = []
    possible_answers_1 = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction_1, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N1):
        thinking1_i, answer1_i = await cot_sc_agents_1[i]([taskInfo], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_1[i].id}, generate possible structure of product 1, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings_1.append(thinking1_i)
        possible_answers_1.append(answer1_i)
    final_decision_agent_1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings_1 + possible_answers_1, "Sub-task 1: Synthesize and choose the most consistent answer for product 1.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1["response"] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = "Sub-task 2: Predict the structure of product 2 arising from tosylhydrazone formation of the C-1 ketone in product 1 with p-toluenesulfonyl hydrazide and HCl."
    N2 = self.max_sc
    cot_sc_agents_2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings_2 = []
    possible_answers_2 = []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction_2, "context": ["user query","thinking1","answer1"], "agent_collaboration": "SC_CoT"}
    for i in range(N2):
        thinking2_i, answer2_i = await cot_sc_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2[i].id}, generate possible structure of product 2, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings_2.append(thinking2_i)
        possible_answers_2.append(answer2_i)
    final_decision_agent_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings_2 + possible_answers_2, "Sub-task 2: Synthesize and choose the most consistent answer for product 2.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2["response"] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = "Sub-task 3: Predict the structure of product 3 resulting from the base-induced Shapiro/Bamford–Stevens type reaction of the tosylhydrazone in product 2 followed by aqueous ammonium chloride quench."
    N3 = self.max_sc
    cot_sc_agents_3 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_thinkings_3 = []
    possible_answers_3 = []
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_instruction_3, "context": ["user query","thinking2","answer2"], "agent_collaboration": "SC_CoT"}
    for i in range(N3):
        thinking3_i, answer3_i = await cot_sc_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_3[i].id}, generate possible structure of product 3, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_thinkings_3.append(thinking3_i)
        possible_answers_3.append(answer3_i)
    final_decision_agent_3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + possible_thinkings_3 + possible_answers_3, "Sub-task 3: Synthesize and choose the most consistent answer for product 3.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3["response"] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    cot_sc_instruction_4 = "Sub-task 4: Predict the structure of product 4 obtained by hydrogenation (Pd/C, H2) of product 3, considering benzyl ether cleavage and double-bond reduction."
    N4 = self.max_sc
    cot_sc_agents_4 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_thinkings_4 = []
    possible_answers_4 = []
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_sc_instruction_4, "context": ["user query","thinking3","answer3"], "agent_collaboration": "SC_CoT"}
    for i in range(N4):
        thinking4_i, answer4_i = await cot_sc_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_4[i].id}, generate possible structure of product 4, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
        possible_thinkings_4.append(thinking4_i)
        possible_answers_4.append(answer4_i)
    final_decision_agent_4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking3, answer3] + possible_thinkings_4 + possible_answers_4, "Sub-task 4: Synthesize and choose the most consistent answer for product 4.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4["response"] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_5 = "Sub-task 5: Compare the predicted structure of product 4 with the four given choices and select the correct one." + debate_instr
    debate_agents_5 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N5 = self.max_round
    all_thinking5 = [[] for _ in range(N5)]
    all_answer5 = [[] for _ in range(N5)]
    subtask_desc5 = {"subtask_id": "subtask_5", "instruction": debate_instruction_5, "context": ["user query","thinking4","answer4"], "agent_collaboration": "Debate"}
    for r in range(N5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5_i, answer5_i = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                thinking5_i, answer5_i = await agent([taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1], debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking5_i.content}; answer: {answer5_i.content}")
            all_thinking5[r].append(thinking5_i)
            all_answer5[r].append(answer5_i)
    final_decision_agent_5 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking5, answer5 = await final_decision_agent_5([taskInfo, thinking4, answer4] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Choose the correct structure of product 4." + final_instr, is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5["response"] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs