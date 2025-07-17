async def forward_153(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Sub-task 1: Extract and summarize all given spectral and mass-spectral data (molecular ion peaks, isotope pattern, IR bands, 1H NMR signals, and candidate structures)."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_instruction, "context": ["user query"], "agent_collaboration": "CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extract and summarize data, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    N = self.max_sc
    sc_instruction2 = "Sub-task 2: Interpret the mass spectrum: determine molecular weight, confirm the presence and number of chlorine atoms via the 156/158 peak ratio, and infer a plausible molecular formula."
    sc_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": sc_instruction2, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    possible_thinkings2 = []
    possible_answers2 = []
    for i in range(N):
        thinking2_i, answer2_i = await sc_agents2[i]([taskInfo, thinking1, answer1], sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents2[i].id}, analyze mass spec, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings2.append(thinking2_i)
        possible_answers2.append(answer2_i)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr2 = "Given all the above thinking and answers, find the most consistent and correct solutions for the problem."
    thinking2, answer2 = await final_decision_agent2(
        [taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2,
        "Sub-task 2: Synthesize and choose the most consistent answer for the problem. " + final_instr2,
        is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision_agent2.id}, synthesize mass spec interpretation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    sc_instruction3 = "Sub-task 3: Interpret IR and 1H NMR data: identify functional groups (e.g., carboxylic acid O–H and C=O) and deduce aromatic substitution pattern (e.g., para-disubstitution from an AA′BB′ system)."
    sc_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": sc_instruction3, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "SC_CoT"}
    possible_thinkings3 = []
    possible_answers3 = []
    for i in range(N):
        thinking3_i, answer3_i = await sc_agents3[i]([taskInfo, thinking1, answer1], sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents3[i].id}, analyze IR/NMR, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_thinkings3.append(thinking3_i)
        possible_answers3.append(answer3_i)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Given all the above thinking and answers, find the most consistent and correct solutions for the problem."
    thinking3, answer3 = await final_decision_agent3(
        [taskInfo, thinking1, answer1] + possible_thinkings3 + possible_answers3,
        "Sub-task 3: Synthesize and choose the most consistent answer for the problem. " + final_instr3,
        is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision_agent3.id}, synthesize IR/NMR interpretation, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction4 = "Sub-task 4: Compare the inferred features (one Cl atom, carboxylic acid, para-disubstituted benzene) against provided candidates and select the best fitting structure." + debate_instr
    debate_agents4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": debate_instruction4, "context": ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents4):
            if r == 0:
                thinking4_i, answer4_i = await agent(
                    [taskInfo, thinking2, answer2, thinking3, answer3],
                    debate_instruction4,
                    r,
                    is_sub_task=True
                )
            else:
                input_infos4 = [taskInfo, thinking2, answer2, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4_i, answer4_i = await agent(
                    input_infos4,
                    debate_instruction4,
                    r,
                    is_sub_task=True
                )
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
            all_thinking4[r].append(thinking4_i)
            all_answer4[r].append(answer4_i)
    final_decision_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr4 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4, answer4 = await final_decision_agent4(
        [taskInfo, thinking2, answer2, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1],
        "Sub-task 4: Select the best structure." + final_instr4,
        is_sub_task=True
    )
    agents.append(f"Final Decision agent {final_decision_agent4.id}, reasoning aggregated debate, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs