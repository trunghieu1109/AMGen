async def forward_155(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    # Stage 0: Extract and summarize core details (SC_CoT)
    cot_sc_instruction0 = "Sub-task 0: Extract and summarize the core details of each reaction and the analytical methods (starting isomers, reagent, workup, and HPLC types) from the query."
    N0 = self.max_sc
    sc_agents0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N0)]
    possible_thinkings0, possible_answers0 = [], []
    subtask_desc0 = {"subtask_id": "subtask_0", "instruction": cot_sc_instruction0, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N0):
        thinking0, answer0 = await sc_agents0[i]([taskInfo], cot_sc_instruction0, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents0[i].id}, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_thinkings0.append(thinking0)
        possible_answers0.append(answer0)
    final_decision0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr0 = "Sub-task 0: Synthesize and choose the most consistent summary for subtask 0. Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking0, answer0 = await final_decision0([taskInfo] + possible_thinkings0 + possible_answers0, final_instr0, is_sub_task=True)
    sub_tasks.append(f"Sub-task 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])
    # Stage 1: Stereochemical outcome (Debate)
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction1 = "Sub-task 1: Determine the stereochemical outcome of epoxidation and acid workup on (E)- and (Z)-oct-4-ene: how many stereocenters form, whether each product is racemic, and diastereomeric relationships." + debate_instr
    debate_agents1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinkings1 = [[] for _ in range(self.max_round)]
    all_answers1 = [[] for _ in range(self.max_round)]
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": debate_instruction1, "context": ["user query", "thinking of subtask 0", "answer of subtask 0"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents1:
            if r == 0:
                thinking1, answer1 = await agent([taskInfo, thinking0, answer0], debate_instruction1, r, is_sub_task=True)
            else:
                inputs1 = [taskInfo, thinking0, answer0] + all_thinkings1[r-1] + all_answers1[r-1]
                thinking1, answer1 = await agent(inputs1, debate_instruction1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinkings1[r].append(thinking1)
            all_answers1[r].append(answer1)
    final_decision1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr1 = "Sub-task 1: Determine stereochemical outcome. Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking1, answer1 = await final_decision1([taskInfo, thinking0, answer0] + all_thinkings1[-1] + all_answers1[-1], final_instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 2: List stereoisomeric products (SC_CoT)
    cot_sc_instruction2 = "Sub-task 2: List all stereoisomeric products arising from both reactions and classify them into diastereomeric sets."
    N2 = self.max_sc
    sc_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2, possible_answers2 = [], []
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": cot_sc_instruction2, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    for i in range(N2):
        thinking2, answer2 = await sc_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents2[i].id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings2.append(thinking2)
        possible_answers2.append(answer2)
    final_decision2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr2 = "Sub-task 2: Synthesize and choose the most consistent list for subtask 2. Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking2, answer2 = await final_decision2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, final_instr2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 3: Predict HPLC peaks (Debate)
    debate_instruction3 = "Sub-task 3: Predict how many peaks each set will give on an achiral reverse-phase HPLC and on a chiral HPLC, then select the matching answer choice." + debate_instr
    debate_agents3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinkings3 = [[] for _ in range(self.max_round)]
    all_answers3 = [[] for _ in range(self.max_round)]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": debate_instruction3, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for agent in debate_agents3:
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction3, r, is_sub_task=True)
            else:
                inputs3 = [taskInfo, thinking2, answer2] + all_thinkings3[r-1] + all_answers3[r-1]
                thinking3, answer3 = await agent(inputs3, debate_instruction3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinkings3[r].append(thinking3)
            all_answers3[r].append(answer3)
    final_decision3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Sub-task 3: Predict HPLC peaks. Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking3, answer3 = await final_decision3([taskInfo, thinking2, answer2] + all_thinkings3[-1] + all_answers3[-1], final_instr3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs