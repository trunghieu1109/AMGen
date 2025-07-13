async def forward_153(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 0: SC_CoT for data extraction
    cot_sc_instruction = "Sub-task 0: Extract and organize all relevant data from the query, including MS peaks, IR absorptions, 1H NMR signals, and the candidate structures."
    N0 = self.max_sc
    cot_sc_agents0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N0)]
    possible_thinkings0 = []
    possible_answers0 = []
    subtask_desc0 = {
        "subtask_id": "subtask_0",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N0):
        thinking0, answer0 = await cot_sc_agents0[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents0[i].id}, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_thinkings0.append(thinking0)
        possible_answers0.append(answer0)
    final_decision0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final0, answer0 = await final_decision0([taskInfo] + possible_thinkings0 + possible_answers0, 
        "Given all the above thinking and answers, synthesize and choose the most consistent extraction of spectral and candidates data.", 
        is_sub_task=True)
    sub_tasks.append(f"Sub-task 0 output: thinking - {final0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": final0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])

    # Sub-task 1: SC_CoT for feature analysis
    cot_sc_instruction = "Sub-task 1: Analyze the extracted data to identify key structural features, such as chlorine isotope pattern, carboxylic OH, and aromatic substitution pattern."
    N1 = self.max_sc
    cot_sc_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query", "thinking of subtask_0", "answer of subtask_0"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N1):
        thinking1, answer1 = await cot_sc_agents1[i]([taskInfo, final0, answer0], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents1[i].id}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings1.append(thinking1)
        possible_answers1.append(answer1)
    final_decision1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final1, answer1 = await final_decision1([taskInfo, final0, answer0] + possible_thinkings1 + possible_answers1,
        "Given all the above thinking and answers, synthesize and choose the most consistent feature analysis.",
        is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {final1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": final1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: SC_CoT for variant generation
    cot_sc_instruction = "Sub-task 2: Generate possible structural variants consistent with the identified features, considering benzoic acid isomers and other candidates."
    N2 = self.max_sc
    cot_sc_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction,
        "context": ["user query", "thinking of subtask_1", "answer of subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2, answer2 = await cot_sc_agents2[i]([taskInfo, final1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents2[i].id}, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_thinkings2.append(thinking2)
        possible_answers2.append(answer2)
    final_decision2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final2, answer2 = await final_decision2([taskInfo, final1, answer1] + possible_thinkings2 + possible_answers2,
        "Given all the above thinking and answers, synthesize and choose the most consistent set of structural variants.",
        is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {final2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": final2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Debate for final evaluation
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction = "Sub-task 3: Evaluate and prioritize the generated structures against all spectral data to choose the most plausible candidate." + debate_instr
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction,
        "context": ["user query", "thinking of subtask_2", "answer of subtask_2"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, final2, answer2], debate_instruction, r, is_sub_task=True)
            else:
                inputs = [taskInfo, final2, answer2] + all_thinking[r-1] + all_answer[r-1]
                thinking3, answer3 = await agent(inputs, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking[r].append(thinking3)
            all_answer[r].append(answer3)
    final_decision3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final3, answer3 = await final_decision3([taskInfo, final2, answer2] + all_thinking[-1] + all_answer[-1],
        "Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {final3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": final3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(final3, answer3, sub_tasks, agents)
    return final_answer, logs