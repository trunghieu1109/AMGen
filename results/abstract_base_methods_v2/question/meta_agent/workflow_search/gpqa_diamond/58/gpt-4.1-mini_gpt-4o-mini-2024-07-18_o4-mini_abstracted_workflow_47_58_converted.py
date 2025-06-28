async def forward_58(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1 = "Sub-task 1: Calculate the initial moles of each species: CH3COOH, HCl, and Ba(OH)2 using their given concentrations and volumes from the query."
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, calculating initial moles of CH3COOH, HCl, Ba(OH)2, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    
    cot_instruction_2 = "Sub-task 2: Calculate the total volume of the combined solution by summing the volumes of all individual solutions (CH3COOH, HCl, Ba(OH)2)."
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_instruction_2,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, calculating total volume, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    
    cot_instruction_3 = "Sub-task 3: Calculate the initial moles of H+ from HCl (strong acid) and OH- from Ba(OH)2 (strong base) before any reaction occurs, using moles from Sub-task 1."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask_1", "answer of subtask_1"],
        "agent_collaboration": "CoT"
    }
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating initial moles of H+ and OH-, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_instruction_4 = "Sub-task 4: Calculate the moles of H+ neutralized by OH- from Ba(OH)2 and determine the net moles of excess H+ or OH- after this neutralization."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask_3", "answer of subtask_3"],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating neutralization and net moles, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])
    
    cot_instruction_5a = "Sub-task 5a: Model the neutralization reaction between any excess OH- (if present) and CH3COOH: calculate moles of CH3COOH neutralized, moles of acetate ion (CH3COO-) formed, and remaining moles of OH- after this reaction using the equation CH3COOH + OH- → CH3COO- + H2O."
    N = self.max_sc
    cot_agents_5a = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_5a = []
    thinkingmapping_5a = {}
    answermapping_5a = {}
    subtask_desc5a = {
        "subtask_id": "subtask_5a",
        "instruction": cot_instruction_5a,
        "context": ["user query", "thinking of subtask_4", "answer of subtask_4", "thinking of subtask_1", "answer of subtask_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking5a, answer5a = await cot_agents_5a[i]([taskInfo, thinking4, answer4, thinking1, answer1], cot_instruction_5a, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_5a[i].id}, modeling neutralization between excess OH- and CH3COOH, thinking: {thinking5a.content}; answer: {answer5a.content}")
        possible_answers_5a.append(answer5a.content)
        thinkingmapping_5a[answer5a.content] = thinking5a
        answermapping_5a[answer5a.content] = answer5a
    answer5a_content = Counter(possible_answers_5a).most_common(1)[0][0]
    thinking5a = thinkingmapping_5a[answer5a_content]
    answer5a = answermapping_5a[answer5a_content]
    sub_tasks.append(f"Sub-task 5a output: thinking - {thinking5a.content}; answer - {answer5a.content}")
    subtask_desc5a['response'] = {"thinking": thinking5a, "answer": answer5a}
    logs.append(subtask_desc5a)
    print("Step 5a: ", sub_tasks[-1])
    
    cot_instruction_5b = "Sub-task 5b: Using the acetic acid dissociation constant (Ka = 1.8×10⁻⁵), calculate the equilibrium concentrations of CH3COOH, CH3COO-, H+, and OH- in the final solution, considering the residual species after neutralization and the total volume."
    cot_agent_5b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5b = {
        "subtask_id": "subtask_5b",
        "instruction": cot_instruction_5b,
        "context": ["user query", "thinking of subtask_5a", "answer of subtask_5a", "thinking of subtask_2", "answer of subtask_2"],
        "agent_collaboration": "CoT"
    }
    thinking5b, answer5b = await cot_agent_5b([taskInfo, thinking5a, answer5a, thinking2, answer2], cot_instruction_5b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5b.id}, calculating equilibrium concentrations with Ka, thinking: {thinking5b.content}; answer: {answer5b.content}")
    sub_tasks.append(f"Sub-task 5b output: thinking - {thinking5b.content}; answer - {answer5b.content}")
    subtask_desc5b['response'] = {"thinking": thinking5b, "answer": answer5b}
    logs.append(subtask_desc5b)
    print("Step 5b: ", sub_tasks[-1])
    
    cot_instruction_6 = "Sub-task 6: Calculate the pH of the final solution based on the equilibrium concentration of H+ (or OH-) obtained from Sub-task 5b."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask_5b", "answer of subtask_5b"],
        "agent_collaboration": "CoT"
    }
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5b, answer5b], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, calculating pH from equilibrium concentrations, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])
    
    cot_instruction_7 = "Sub-task 7: Verify the chemical consistency of the calculated equilibrium concentrations and pH, checking for any contradictions or errors; if inconsistencies are found, flag for re-evaluation of previous subtasks."
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking5b, answer5b, thinking6, answer6]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", "thinking of subtask_5b", "answer of subtask_5b", "thinking of subtask_6", "answer of subtask_6"],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, verifying chemical consistency, thinking: {thinking7.content}; answer: {answer7.content}")
    for i in range(N_max_7):
        feedback7, correct7 = await critic_agent_7([taskInfo, thinking7, answer7], "Please review the chemical consistency of equilibrium concentrations and pH, and provide limitations or confirm correctness.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}")
        if correct7.content == "True":
            break
        cot_inputs_7.extend([thinking7, answer7, feedback7])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining chemical consistency verification, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])
    
    cot_instruction_8 = "Sub-task 8: Compare the calculated pH with the given multiple-choice options and select the correct answer choice (A, B, C, or D)."
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", "thinking of subtask_6", "answer of subtask_6", "thinking of subtask_7", "answer of subtask_7"],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking6, answer6, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, selecting correct pH choice, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs
