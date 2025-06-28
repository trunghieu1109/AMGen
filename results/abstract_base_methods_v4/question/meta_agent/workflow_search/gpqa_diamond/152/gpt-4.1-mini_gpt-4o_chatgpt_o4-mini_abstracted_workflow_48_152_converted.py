async def forward_152(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Analyze the general mechanism of Michael addition reactions, emphasizing nucleophilic attack at the β-carbon of α,β-unsaturated carbonyl compounds, formation of resonance-stabilized intermediates, and the role of enolate ions as nucleophiles, to establish foundational understanding for subsequent reaction-specific analysis."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing Michael addition mechanism, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_sc_instruction_2 = "Sub-task 2: Identify and characterize all reactants involved in the three given Michael addition reactions, explicitly distinguishing tautomeric forms where relevant (e.g., canonical keto form cyclohexane-1,3-dione vs. enol tautomer 2-hydroxycyclohexane-1,3-dione for reactant C), including their functional groups, nucleophilic/electrophilic roles, and reaction conditions (catalysts, solvents), based on the understanding from Sub-task 1. Generate multiple candidate identifications and select the most chemically accurate one emphasizing canonical keto forms."
    N = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_2 = []
    thinkingmapping_2 = {}
    answermapping_2 = {}
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, identifying reactants and roles with tautomer distinction, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2.content)
        thinkingmapping_2[answer2.content] = thinking2
        answermapping_2[answer2.content] = answer2
    most_common_answer_2 = Counter(possible_answers_2).most_common(1)[0][0]
    thinking2 = thinkingmapping_2[most_common_answer_2]
    answer2 = answermapping_2[most_common_answer_2]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_reflect_instruction_3 = "Sub-task 3: Create a consolidated summary table listing each reaction (A, B, C) with their identified reactants, reagents, and expected products (names and structures), referencing canonical forms and clarifying tautomeric distinctions to ensure consistency for downstream subtasks, based on Sub-task 2 outputs."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_reflect_instruction_3, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, creating consolidated summary table, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_sc_instruction_4 = "Sub-task 4: For reaction A (dimethyl malonate + methyl (E)-3-(p-tolyl)acrylate + NaOEt/EtOH), predict the major product by applying the Michael addition mechanism, considering nucleophile and electrophile roles, resonance stabilization, and reaction conditions; determine the product’s correct chemical name and structure, based on Sub-task 3 outputs."
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_4 = []
    thinkingmapping_4 = {}
    answermapping_4 = {}
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3, answer3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, predicting product of reaction A, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4.content)
        thinkingmapping_4[answer4.content] = thinking4
        answermapping_4[answer4.content] = answer4
    most_common_answer_4 = Counter(possible_answers_4).most_common(1)[0][0]
    thinking4 = thinkingmapping_4[most_common_answer_4]
    answer4 = answermapping_4[most_common_answer_4]
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_sc_instruction_5 = "Sub-task 5: For reaction B (1-(cyclohex-1-en-1-yl)piperidine + (E)-but-2-enenitrile + MeOH/H3O+), predict the major product by applying the Michael addition mechanism and subsequent transformations under acidic conditions; determine the product’s correct chemical name and structure, based on Sub-task 3 outputs."
    cot_agents_5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_5 = []
    thinkingmapping_5 = {}
    answermapping_5 = {}
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction_5,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5, answer5 = await cot_agents_5[i]([taskInfo, thinking3, answer3], cot_sc_instruction_5, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5[i].id}, predicting product of reaction B, thinking: {thinking5.content}; answer: {answer5.content}")
        possible_answers_5.append(answer5.content)
        thinkingmapping_5[answer5.content] = thinking5
        answermapping_5[answer5.content] = answer5
    most_common_answer_5 = Counter(possible_answers_5).most_common(1)[0][0]
    thinking5 = thinkingmapping_5[most_common_answer_5]
    answer5 = answermapping_5[most_common_answer_5]
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_instruction_6a = "Sub-task 6a: Accurately identify reactant C based on the given product name (2-(3-oxobutyl)cyclohexane-1,3-dione), explicitly distinguishing the canonical keto form cyclohexane-1,3-dione from its enol tautomer, and provide detailed structural and reactivity information to support correct identification, based on Sub-task 3 outputs."
    cot_agent_6a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
    subtask_desc6a = {
        "subtask_id": "subtask_6a",
        "instruction": cot_instruction_6a,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking6a, answer6a = await cot_agent_6a([taskInfo, thinking3, answer3], cot_instruction_6a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6a.id}, identifying reactant C with tautomer distinction, thinking: {thinking6a.content}; answer: {answer6a.content}")
    sub_tasks.append(f"Sub-task 6a output: thinking - {thinking6a.content}; answer - {answer6a.content}")
    subtask_desc6a['response'] = {"thinking": thinking6a, "answer": answer6a}
    logs.append(subtask_desc6a)
    print("Step 6a: ", sub_tasks[-1])
    
    cot_sc_instruction_6b = "Sub-task 6b: Using the correctly identified reactant C from Sub-task 6a, predict the major product of reaction C (C + but-3-en-2-one + KOH/H2O) by applying the Michael addition mechanism and subsequent transformations; determine the product’s correct chemical name and structure, based on Sub-task 6a outputs."
    cot_agents_6b = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_6b = []
    thinkingmapping_6b = {}
    answermapping_6b = {}
    subtask_desc6b = {
        "subtask_id": "subtask_6b",
        "instruction": cot_sc_instruction_6b,
        "context": ["user query", "thinking of subtask 6a", "answer of subtask 6a"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking6b, answer6b = await cot_agents_6b[i]([taskInfo, thinking6a, answer6a], cot_sc_instruction_6b, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_6b[i].id}, predicting product of reaction C, thinking: {thinking6b.content}; answer: {answer6b.content}")
        possible_answers_6b.append(answer6b.content)
        thinkingmapping_6b[answer6b.content] = thinking6b
        answermapping_6b[answer6b.content] = answer6b
    most_common_answer_6b = Counter(possible_answers_6b).most_common(1)[0][0]
    thinking6b = thinkingmapping_6b[most_common_answer_6b]
    answer6b = answermapping_6b[most_common_answer_6b]
    sub_tasks.append(f"Sub-task 6b output: thinking - {thinking6b.content}; answer - {answer6b.content}")
    subtask_desc6b['response'] = {"thinking": thinking6b, "answer": answer6b}
    logs.append(subtask_desc6b)
    print("Step 6b: ", sub_tasks[-1])
    
    debate_instruction_7a = "Sub-task 7a: Verify the consistency between predicted products and reactants for each reaction (A, B, and C) individually, ensuring chemical logic, correct nucleophile-electrophile pairing, and proper naming conventions, based on Sub-tasks 4, 5, and 6b outputs."
    debate_agents_7a = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7a = self.max_round
    all_thinking7a = [[] for _ in range(N_max_7a)]
    all_answer7a = [[] for _ in range(N_max_7a)]
    subtask_desc7a = {
        "subtask_id": "subtask_7a",
        "instruction": debate_instruction_7a,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4", "thinking of subtask 5", "answer of subtask 5", "thinking of subtask 6b", "answer of subtask 6b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_7a):
        for i, agent in enumerate(debate_agents_7a):
            if r == 0:
                thinking7a, answer7a = await agent([taskInfo, thinking4, answer4, thinking5, answer5, thinking6b, answer6b], debate_instruction_7a, r, is_sub_task=True)
            else:
                input_infos_7a = [taskInfo, thinking4, answer4, thinking5, answer5, thinking6b, answer6b] + all_thinking7a[r-1] + all_answer7a[r-1]
                thinking7a, answer7a = await agent(input_infos_7a, debate_instruction_7a, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, verifying reactant-product consistency, thinking: {thinking7a.content}; answer: {answer7a.content}")
            all_thinking7a[r].append(thinking7a)
            all_answer7a[r].append(answer7a)
    final_decision_agent_7a = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7a, answer7a = await final_decision_agent_7a([taskInfo] + all_thinking7a[-1] + all_answer7a[-1], "Sub-task 7a: Make final decision on individual reaction consistency.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting consistent individual reactions, thinking: {thinking7a.content}; answer: {answer7a.content}")
    sub_tasks.append(f"Sub-task 7a output: thinking - {thinking7a.content}; answer - {answer7a.content}")
    subtask_desc7a['response'] = {"thinking": thinking7a, "answer": answer7a}
    logs.append(subtask_desc7a)
    print("Step 7a: ", sub_tasks[-1])
    
    debate_instruction_7b = "Sub-task 7b: Cross-check all three reactions’ predicted reactants and products against the multiple-choice options, analyzing structural and nomenclature consistency to identify the choice that correctly matches all three reaction outcomes, based on Sub-task 7a output."
    debate_agents_7b = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7b = self.max_round
    all_thinking7b = [[] for _ in range(N_max_7b)]
    all_answer7b = [[] for _ in range(N_max_7b)]
    subtask_desc7b = {
        "subtask_id": "subtask_7b",
        "instruction": debate_instruction_7b,
        "context": ["user query", "thinking of subtask 7a", "answer of subtask 7a"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_7b):
        for i, agent in enumerate(debate_agents_7b):
            if r == 0:
                thinking7b, answer7b = await agent([taskInfo, thinking7a, answer7a], debate_instruction_7b, r, is_sub_task=True)
            else:
                input_infos_7b = [taskInfo, thinking7a, answer7a] + all_thinking7b[r-1] + all_answer7b[r-1]
                thinking7b, answer7b = await agent(input_infos_7b, debate_instruction_7b, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, cross-checking choices, thinking: {thinking7b.content}; answer: {answer7b.content}")
            all_thinking7b[r].append(thinking7b)
            all_answer7b[r].append(answer7b)
    final_decision_agent_7b = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7b, answer7b = await final_decision_agent_7b([taskInfo] + all_thinking7b[-1] + all_answer7b[-1], "Sub-task 7b: Make final decision on correct multiple-choice option.", is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct multiple-choice option, thinking: {thinking7b.content}; answer: {answer7b.content}")
    sub_tasks.append(f"Sub-task 7b output: thinking - {thinking7b.content}; answer - {answer7b.content}")
    subtask_desc7b['response'] = {"thinking": thinking7b, "answer": answer7b}
    logs.append(subtask_desc7b)
    print("Step 7b: ", sub_tasks[-1])
    
    debate_instruction_7c = "Sub-task 7c: Perform a final integrated consistency check and reflective debate phase where agents critically evaluate and reconcile any discrepancies between predicted results and multiple-choice options, ensuring alignment of reactants, products, and chemical logic before selecting the final answer."
    debate_agents_7c = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_7c = self.max_round
    all_thinking7c = [[] for _ in range(N_max_7c)]
    all_answer7c = [[] for _ in range(N_max_7c)]
    subtask_desc7c = {
        "subtask_id": "subtask_7c",
        "instruction": debate_instruction_7c,
        "context": ["user query", "thinking of subtask 7b", "answer of subtask 7b"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_7c):
        for i, agent in enumerate(debate_agents_7c):
            if r == 0:
                thinking7c, answer7c = await agent([taskInfo, thinking7b, answer7b], debate_instruction_7c, r, is_sub_task=True)
            else:
                input_infos_7c = [taskInfo, thinking7b, answer7b] + all_thinking7c[r-1] + all_answer7c[r-1]
                thinking7c, answer7c = await agent(input_infos_7c, debate_instruction_7c, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, final reflective debate, thinking: {thinking7c.content}; answer: {answer7c.content}")
            all_thinking7c[r].append(thinking7c)
            all_answer7c[r].append(answer7c)
    final_decision_agent_7c = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking7c, answer7c = await final_decision_agent_7c([taskInfo] + all_thinking7c[-1] + all_answer7c[-1], "Sub-task 7c: Make final integrated decision on the correct answer.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final integrated decision, thinking: {thinking7c.content}; answer: {answer7c.content}")
    sub_tasks.append(f"Sub-task 7c output: thinking - {thinking7c.content}; answer - {answer7c.content}")
    subtask_desc7c['response'] = {"thinking": thinking7c, "answer": answer7c}
    logs.append(subtask_desc7c)
    print("Step 7c: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking7c, answer7c, sub_tasks, agents)
    return final_answer, logs