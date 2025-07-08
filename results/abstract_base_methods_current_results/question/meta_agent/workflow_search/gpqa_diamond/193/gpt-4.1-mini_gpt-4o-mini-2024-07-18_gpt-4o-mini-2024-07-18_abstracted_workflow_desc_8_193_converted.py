async def forward_193(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Enumerate all possible spin configurations for the three spins S1, S2, and S3, where each spin can be +1 or -1. Explicitly list each configuration as a tuple (S1, S2, S3)."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, enumerating all spin configurations, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {
        "thinking": thinking1,
        "answer": answer1
    }
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    spin_configurations = []
    try:
        lines = answer1.content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('(') and line.endswith(')'):
                parts = line[1:-1].split(',')
                if len(parts) == 3:
                    s = tuple(int(p.strip()) for p in parts)
                    spin_configurations.append(s)
    except:
        spin_configurations = []
    
    if len(spin_configurations) != 8:
        spin_configurations = [
            (+1, +1, +1),
            (+1, +1, -1),
            (+1, -1, +1),
            (+1, -1, -1),
            (-1, +1, +1),
            (-1, +1, -1),
            (-1, -1, +1),
            (-1, -1, -1)
        ]
    
    energies = {}
    subtask_2_ids = ['subtask_2a','subtask_2b','subtask_2c','subtask_2d','subtask_2e','subtask_2f','subtask_2g','subtask_2h']
    for idx, config in enumerate(spin_configurations):
        cot_sc_instruction_2 = f"Sub-task 2{chr(ord('a')+idx)}: Calculate the energy E for the spin configuration {config} using the formula E = -J (S1*S2 + S1*S3 + S2*S3). Restate the sign convention clearly before calculation."
        N = self.max_sc
        cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
        possible_answers_2 = []
        thinkingmapping_2 = {}
        answermapping_2 = {}
        subtask_desc2 = {
            "subtask_id": subtask_2_ids[idx],
            "instruction": cot_sc_instruction_2,
            "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
            "agent_collaboration": "SC_CoT"
        }
        for i in range(N):
            thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
            agents.append(f"CoT-SC agent {cot_agents_2[i].id}, calculating energy for configuration {config}, thinking: {thinking2.content}; answer: {answer2.content}")
            possible_answers_2.append(answer2.content)
            thinkingmapping_2[answer2.content] = thinking2
            answermapping_2[answer2.content] = answer2
        answer2_content = Counter(possible_answers_2).most_common(1)[0][0]
        thinking2 = thinkingmapping_2[answer2_content]
        answer2 = answermapping_2[answer2_content]
        sub_tasks.append(f"Sub-task 2{chr(ord('a')+idx)} output: thinking - {thinking2.content}; answer - {answer2.content}")
        subtask_desc2['response'] = {
            "thinking": thinking2,
            "answer": answer2
        }
        logs.append(subtask_desc2)
        print(f"Step 2{chr(ord('a')+idx)}: ", sub_tasks[-1])
        energies[config] = answer2_content
    
    cot_instruction_2_verification = "Sub-task 2_verification: Perform a self-consistency check by comparing all energy calculations from subtasks 2a to 2h to ensure no sign or arithmetic errors exist. Confirm energies match expected symmetry and values."
    cot_agent_2_verification = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2_verification = {
        "subtask_id": "subtask_2_verification",
        "instruction": cot_instruction_2_verification,
        "context": ["user query"] + [e for e in energies.values()],
        "agent_collaboration": "CoT"
    }
    thinking2v, answer2v = await cot_agent_2_verification([taskInfo] + list(energies.values()), cot_instruction_2_verification, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_verification.id}, verifying energies self-consistency, thinking: {thinking2v.content}; answer: {answer2v.content}")
    sub_tasks.append(f"Sub-task 2_verification output: thinking - {thinking2v.content}; answer - {answer2v.content}")
    subtask_desc2_verification['response'] = {
        "thinking": thinking2v,
        "answer": answer2v
    }
    logs.append(subtask_desc2_verification)
    print("Step 2_verification: ", sub_tasks[-1])
    
    cot_sc_instruction_3 = "Sub-task 3: Compute the Boltzmann factor e^(-βE) for each spin configuration using the verified energies from subtask_2_verification, where β = 1/(kT)."
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    boltzmann_factors = {}
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 2_verification", "answer of subtask 2_verification"],
        "agent_collaboration": "SC_CoT"
    }
    for config, energy_str in energies.items():
        possible_answers_3 = []
        thinkingmapping_3 = {}
        answermapping_3 = {}
        for i in range(N):
            thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking2v, answer2v, energy_str], cot_sc_instruction_3, is_sub_task=True)
            agents.append(f"CoT-SC agent {cot_agents_3[i].id}, computing Boltzmann factor for energy {energy_str}, thinking: {thinking3.content}; answer: {answer3.content}")
            possible_answers_3.append(answer3.content)
            thinkingmapping_3[answer3.content] = thinking3
            answermapping_3[answer3.content] = answer3
        answer3_content = Counter(possible_answers_3).most_common(1)[0][0]
        thinking3 = thinkingmapping_3[answer3_content]
        answer3 = answermapping_3[answer3_content]
        boltzmann_factors[config] = answer3_content
    sub_tasks.append(f"Sub-task 3 output: Boltzmann factors computed for all configurations.")
    subtask_desc3['response'] = {
        "thinking": thinking3,
        "answer": answer3
    }
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_instruction_4 = "Sub-task 4: Sum all Boltzmann factors from Sub-task 3 to obtain the partition function Z of the system. Verify that Z(β=0) = 8 as a consistency check."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo] + list(boltzmann_factors.values()), cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, summing Boltzmann factors to get partition function Z, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {
        "thinking": thinking4,
        "answer": answer4
    }
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instruction_5 = "Sub-task 5: Match the computed partition function Z with the given choices to identify the correct expression. Provide a clear logical explanation connecting the computed Z to the selected choice."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, matching partition function with choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct partition function expression.", is_sub_task=True)
    agents.append(f"Final Decision agent, deciding correct partition function, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {
        "thinking": thinking5,
        "answer": answer5
    }
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer, logs
