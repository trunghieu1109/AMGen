async def forward_157(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction = "Sub-task 1: Extract and summarize the molecular pathway of the transcription factor (phosphorylation → dimerization → nuclear translocation → transcription) and define the nature of mutations X and Y."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting and summarizing pathway, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    debate_instr = "Sub-task 2: Assess the effect of a heterozygous dominant-negative mutation in the dimerization domain on wild-type/mutant subunit interactions and functional dimer formation. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking2 = [[] for _ in range(self.max_round)]
    all_answer2 = [[] for _ in range(self.max_round)]
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": debate_instr, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2_temp, answer2_temp = await agent([taskInfo, thinking1, answer1], debate_instr, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
                thinking2_temp, answer2_temp = await agent(inputs, debate_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2_temp.content}; answer: {answer2_temp.content}")
            all_thinking2[r].append(thinking2_temp)
            all_answer2[r].append(answer2_temp)
    final_instr2 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1], "Sub-task 2: Assess the effect of a heterozygous dominant-negative mutation in the dimerization domain on wild-type/mutant subunit interactions and functional dimer formation. " + final_instr2, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    cot_sc_inst3 = "Sub-task 3: Based on the output from Sub-task 2, evaluate the likely molecular mechanism by which the mutant Y subunit poisons the wild-type protein."
    N3 = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_think3 = []
    possible_ans3 = []
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_inst3, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "SC_CoT"}
    for i in range(N3):
        thinking3_temp, answer3_temp = await cot_agents_3[i]([taskInfo, thinking2, answer2], cot_sc_inst3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, evaluating molecular mechanism, thinking: {thinking3_temp.content}; answer: {answer3_temp.content}")
        possible_think3.append(thinking3_temp)
        possible_ans3.append(answer3_temp)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + possible_think3 + possible_ans3, "Sub-task 3: Synthesize and choose the most consistent molecular mechanism for the dominant-negative effect.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    cot_sc_inst4 = "Sub-task 4: Map the deduced molecular mechanism of mutation Y onto the four answer choices and select the phenotype most consistent with a dominant-negative dimerization defect."
    N4 = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_think4 = []
    possible_ans4 = []
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_sc_inst4, "context": ["user query", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "SC_CoT"}
    for i in range(N4):
        thinking4_temp, answer4_temp = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_inst4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, mapping mechanism to choices, thinking: {thinking4_temp.content}; answer: {answer4_temp.content}")
        possible_think4.append(thinking4_temp)
        possible_ans4.append(answer4_temp)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking3, answer3] + possible_think4 + possible_ans4, "Sub-task 4: Synthesize and select the most consistent phenotype mapping.", is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent_4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs