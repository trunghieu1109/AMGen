async def forward_10(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_agents_sc = []
    N_sc = self.max_sc
    
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    
    stage_1_subtasks = [
        {
            "id": "subtask_1",
            "objective": "Analyze the molecular biology concepts related to programmed ribosomal frameshifting in SARS-CoV-2, including the mechanism involving slippery nucleotides and pseudoknots, and compare the conformation of SARS-CoV-2 frameshifting with that of SARS-CoV.",
            "agent": cot_agent_1
        },
        {
            "id": "subtask_2",
            "objective": "Examine the role and molecular function of the SARS-CoV-2 nsp10/nsp14-ExoN complex, focusing on its heterodimer formation, mismatch repair mechanism, and the interaction between nsp14s ExoN domain and nsp10 in preventing dsRNA breakdown.",
            "agent": cot_agent_2
        },
        {
            "id": "subtask_3",
            "objective": "Investigate the apoptotic pathways triggered by SARS-CoV-2 ORF3a, specifically the activation of caspase-8, the involvement of the extrinsic apoptotic pathway via death receptors, and the role of Bcl-2 in the mitochondrial pathway to understand the mechanism of apoptosis induction.",
            "agent": cot_agent_3
        },
        {
            "id": "subtask_4",
            "objective": "Analyze the relationship between the rate of programmed -1 ribosomal frameshifting and the number of conformations adopted by the pseudoknot structures in SARS-CoV and SARS-CoV-2, including the behavior of these pseudoknots under tension and their correlation with frameshifting rates.",
            "agent": cot_agent_4
        }
    ]
    
    for subtask in stage_1_subtasks:
        cot_instruction = f"{subtask['id'].capitalize()}: {subtask['objective']}"
        subtask_desc = {
            "subtask_id": subtask['id'],
            "instruction": cot_instruction,
            "context": ["user query"],
            "agent_collaboration": "CoT"
        }
        thinking, answer = await subtask['agent']([taskInfo], cot_instruction, is_sub_task=True)
        agents.append(f"CoT agent {subtask['agent'].id}, {subtask['id']} reasoning, thinking: {thinking.content}; answer: {answer.content}")
        sub_tasks.append(f"{subtask['id'].capitalize()} output: thinking - {thinking.content}; answer - {answer.content}")
        subtask_desc['response'] = {
            "thinking": thinking,
            "answer": answer
        }
        logs.append(subtask_desc)
        print(f"Step {len(sub_tasks)}: ", sub_tasks[-1])
    
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    N = self.max_sc
    possible_answers = []
    thinkingmapping = {}
    answermapping = {}
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": "Evaluate the correctness of each statement (from subtasks 1 to 4) based on current molecular biology knowledge of SARS-CoV-2, identifying any inaccuracies or inconsistencies.",
        "context": ["user query"] + [st.split(' output: ')[1] for st in sub_tasks],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5, answer5 = await cot_agent_5([taskInfo] + [logs[j]['response']['thinking'] for j in range(4)] + [logs[j]['response']['answer'] for j in range(4)], subtask_desc5['instruction'], is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agent_5.id}, evaluation round {i}, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers.append(answer5.content)
        thinkingmapping[answer5.content] = thinking5
        answermapping[answer5.content] = answer5
    answer5_content = Counter(possible_answers).most_common(1)[0][0]
    thinking5 = thinkingmapping[answer5_content]
    answer5 = answermapping[answer5_content]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking6 = [[] for _ in range(N_max)]
    all_answer6 = [[] for _ in range(N_max)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": "Determine which statement among the given choices is incorrect by contrasting the evaluated facts from subtask 5, and select the corresponding letter choice (A, B, C, or D) as the final answer.",
        "context": ["user query", thinking5, answer5],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], subtask_desc6['instruction'], r, is_sub_task=True)
            else:
                input_infos = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos, subtask_desc6['instruction'], r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the incorrect statement letter choice.", is_sub_task=True)
    agents.append(f"Final Decision agent, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {
        "thinking": thinking6,
        "answer": answer6
    }
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs
