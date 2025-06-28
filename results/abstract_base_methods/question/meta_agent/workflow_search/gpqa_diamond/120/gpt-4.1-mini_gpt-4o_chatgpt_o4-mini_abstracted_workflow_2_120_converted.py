async def forward_120(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction_1a = "Sub-task 1a: Generate a detailed skeletal drawing or adjacency table of (1R,3R,4R,6S)-1,3,4-trimethyl-7-oxabicyclo[4.1.0]heptane, explicitly showing all carbons, the oxygen bridge, and substituents with stereochemistry."
    cot_agent_1a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1a = {
        "subtask_id": "subtask_1a",
        "instruction": cot_instruction_1a,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking_1a, answer_1a = await cot_agent_1a([taskInfo], cot_instruction_1a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1a.id}, generating skeletal drawing and adjacency table, thinking: {thinking_1a.content}; answer: {answer_1a.content}")
    sub_tasks.append(f"Sub-task 1a output: thinking - {thinking_1a.content}; answer - {answer_1a.content}")
    subtask_desc_1a['response'] = {"thinking": thinking_1a, "answer": answer_1a}
    logs.append(subtask_desc_1a)
    print("Step 1a: ", sub_tasks[-1])
    
    cot_instruction_1b = "Sub-task 1b: Label each carbon atom in the bicyclic system with its substituents and stereochemical configuration, following IUPAC numbering conventions for bicyclo[4.1.0]heptane systems, based on output from Sub-task 1a."
    cot_agent_1b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1b = {
        "subtask_id": "subtask_1b",
        "instruction": cot_instruction_1b,
        "context": ["user query", "thinking of subtask 1a", "answer of subtask 1a"],
        "agent_collaboration": "CoT"
    }
    thinking_1b, answer_1b = await cot_agent_1b([taskInfo, thinking_1a, answer_1a], cot_instruction_1b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1b.id}, labeling carbons with substituents and stereochemistry, thinking: {thinking_1b.content}; answer: {answer_1b.content}")
    sub_tasks.append(f"Sub-task 1b output: thinking - {thinking_1b.content}; answer - {answer_1b.content}")
    subtask_desc_1b['response'] = {"thinking": thinking_1b, "answer": answer_1b}
    logs.append(subtask_desc_1b)
    print("Step 1b: ", sub_tasks[-1])
    
    cot_instruction_1c = "Sub-task 1c: Identify and confirm the epoxide ring carbons by analyzing the bicyclic structure and oxygen bridge position, clarifying which two carbons form the epoxide ring, based on outputs from Sub-task 1b."
    cot_agent_1c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_1c = {
        "subtask_id": "subtask_1c",
        "instruction": cot_instruction_1c,
        "context": ["user query", "thinking of subtask 1b", "answer of subtask 1b"],
        "agent_collaboration": "CoT"
    }
    thinking_1c, answer_1c = await cot_agent_1c([taskInfo, thinking_1b, answer_1b], cot_instruction_1c, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1c.id}, identifying epoxide ring carbons, thinking: {thinking_1c.content}; answer: {answer_1c.content}")
    sub_tasks.append(f"Sub-task 1c output: thinking - {thinking_1c.content}; answer - {answer_1c.content}")
    subtask_desc_1c['response'] = {"thinking": thinking_1c, "answer": answer_1c}
    logs.append(subtask_desc_1c)
    print("Step 1c: ", sub_tasks[-1])
    
    cot_sc_instruction_1d = "Sub-task 1d: Perform a self-consistency check (SC CoT) or debate among agents to verify the correct epoxide bridge location and numbering, comparing alternative epoxide positions against the IUPAC name and substitution pattern to ensure accuracy, based on outputs from Sub-task 1c."
    N_sc_1d = self.max_sc
    cot_agents_1d = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1d)]
    possible_answers_1d = []
    thinkingmapping_1d = {}
    answermapping_1d = {}
    subtask_desc_1d = {
        "subtask_id": "subtask_1d",
        "instruction": cot_sc_instruction_1d,
        "context": ["user query", "thinking of subtask 1c", "answer of subtask 1c"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1d):
        thinking_1d, answer_1d = await cot_agents_1d[i]([taskInfo, thinking_1c, answer_1c], cot_sc_instruction_1d, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1d[i].id}, verifying epoxide bridge location, thinking: {thinking_1d.content}; answer: {answer_1d.content}")
        possible_answers_1d.append(answer_1d.content)
        thinkingmapping_1d[answer_1d.content] = thinking_1d
        answermapping_1d[answer_1d.content] = answer_1d
    answer_1d_content = Counter(possible_answers_1d).most_common(1)[0][0]
    thinking_1d = thinkingmapping_1d[answer_1d_content]
    answer_1d = answermapping_1d[answer_1d_content]
    sub_tasks.append(f"Sub-task 1d output: thinking - {thinking_1d.content}; answer - {answer_1d.content}")
    subtask_desc_1d['response'] = {"thinking": thinking_1d, "answer": answer_1d}
    logs.append(subtask_desc_1d)
    print("Step 1d: ", sub_tasks[-1])
    
    cot_instruction_2a = "Sub-task 2a: Analyze steric hindrance at each epoxide carbon by counting substituents and considering their spatial orientation, including ring strain and bridgehead effects, to identify the less hindered carbon for nucleophilic attack, based on verified epoxide structure from Sub-task 1d."
    cot_agent_2a = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2a = {
        "subtask_id": "subtask_2a",
        "instruction": cot_instruction_2a,
        "context": ["user query", "thinking of subtask 1d", "answer of subtask 1d"],
        "agent_collaboration": "CoT"
    }
    thinking_2a, answer_2a = await cot_agent_2a([taskInfo, thinking_1d, answer_1d], cot_instruction_2a, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2a.id}, analyzing steric hindrance at epoxide carbons, thinking: {thinking_2a.content}; answer: {answer_2a.content}")
    sub_tasks.append(f"Sub-task 2a output: thinking - {thinking_2a.content}; answer - {answer_2a.content}")
    subtask_desc_2a['response'] = {"thinking": thinking_2a, "answer": answer_2a}
    logs.append(subtask_desc_2a)
    print("Step 2a: ", sub_tasks[-1])
    
    cot_instruction_2b = "Sub-task 2b: Evaluate electronic and ring strain factors influencing nucleophilic attack site preference on the epoxide ring, referencing literature precedents for organocuprate reactions with bicyclic epoxides, based on outputs from Sub-task 1d."
    cot_agent_2b = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_2b = {
        "subtask_id": "subtask_2b",
        "instruction": cot_instruction_2b,
        "context": ["user query", "thinking of subtask 1d", "answer of subtask 1d"],
        "agent_collaboration": "CoT"
    }
    thinking_2b, answer_2b = await cot_agent_2b([taskInfo, thinking_1d, answer_1d], cot_instruction_2b, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2b.id}, evaluating electronic and ring strain factors, thinking: {thinking_2b.content}; answer: {answer_2b.content}")
    sub_tasks.append(f"Sub-task 2b output: thinking - {thinking_2b.content}; answer - {answer_2b.content}")
    subtask_desc_2b['response'] = {"thinking": thinking_2b, "answer": answer_2b}
    logs.append(subtask_desc_2b)
    print("Step 2b: ", sub_tasks[-1])
    
    cot_reflect_instruction_2c = "Sub-task 2c: Conduct a reflexion or debate step to reconcile steric, electronic, and ring strain analyses and finalize the identification of the less hindered carbon for nucleophilic attack by Me2CuLi, based on outputs from Sub-tasks 2a and 2b."
    cot_agent_2c = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_2c = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_2c = self.max_round
    cot_inputs_2c = [taskInfo, thinking_2a, answer_2a, thinking_2b, answer_2b]
    subtask_desc_2c = {
        "subtask_id": "subtask_2c",
        "instruction": cot_reflect_instruction_2c,
        "context": ["user query", "thinking of subtask 2a", "answer of subtask 2a", "thinking of subtask 2b", "answer of subtask 2b"],
        "agent_collaboration": "Reflexion"
    }
    thinking_2c, answer_2c = await cot_agent_2c(cot_inputs_2c, cot_reflect_instruction_2c, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2c.id}, reconciling analyses to finalize less hindered carbon, thinking: {thinking_2c.content}; answer: {answer_2c.content}")
    for i in range(N_max_2c):
        feedback, correct = await critic_agent_2c([taskInfo, thinking_2c, answer_2c], "please review the identification of the less hindered carbon for nucleophilic attack and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2c.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2c.extend([thinking_2c, answer_2c, feedback])
        thinking_2c, answer_2c = await cot_agent_2c(cot_inputs_2c, cot_reflect_instruction_2c, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2c.id}, refining less hindered carbon identification, thinking: {thinking_2c.content}; answer: {answer_2c.content}")
    sub_tasks.append(f"Sub-task 2c output: thinking - {thinking_2c.content}; answer - {answer_2c.content}")
    subtask_desc_2c['response'] = {"thinking": thinking_2c, "answer": answer_2c}
    logs.append(subtask_desc_2c)
    print("Step 2c: ", sub_tasks[-1])
    
    cot_instruction_3 = "Sub-task 3: Determine the stereochemical outcome of nucleophilic attack at the identified less hindered epoxide carbon, applying the principle of inversion of configuration at the carbon where nucleophile adds, based on output from Sub-task 2c."
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking_2c, answer_2c]
    subtask_desc_3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_instruction_3,
        "context": ["user query", "thinking of subtask 2c", "answer of subtask 2c"],
        "agent_collaboration": "Reflexion"
    }
    thinking_3, answer_3 = await cot_agent_3(cot_inputs_3, cot_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, determining stereochemical outcome, thinking: {thinking_3.content}; answer: {answer_3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking_3, answer_3], "please review the stereochemical inversion determination and provide its limitations.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking_3, answer_3, feedback])
        thinking_3, answer_3 = await cot_agent_3(cot_inputs_3, cot_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining stereochemical outcome, thinking: {thinking_3.content}; answer: {answer_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_3.content}; answer - {answer_3.content}")
    subtask_desc_3['response'] = {"thinking": thinking_3, "answer": answer_3}
    logs.append(subtask_desc_3)
    print("Step 3: ", sub_tasks[-1])
    
    cot_instruction_4 = "Sub-task 4: Predict the ring-opened product structure incorporating the methyl group from Me2CuLi at the less hindered carbon with correct stereochemistry, and assign the resulting stereochemical descriptors consistent with the bicyclic to cyclohexanol transformation, based on output from Sub-task 3."
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction_4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "CoT"
    }
    thinking_4, answer_4 = await cot_agent_4([taskInfo, thinking_3, answer_3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, predicting ring-opened product, thinking: {thinking_4.content}; answer: {answer_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_4.content}; answer - {answer_4.content}")
    subtask_desc_4['response'] = {"thinking": thinking_4, "answer": answer_4}
    logs.append(subtask_desc_4)
    print("Step 4: ", sub_tasks[-1])
    
    debate_instruction_5 = "Sub-task 5: Convert the predicted ring-opened intermediate into the corresponding cyclohexanol product, ensuring correct numbering and stereochemical assignments that match the format of the multiple-choice options, and compare the predicted product with the given choices to identify the correct product, based on output from Sub-task 4."
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking_5 = [[] for _ in range(N_max_5)]
    all_answer_5 = [[] for _ in range(N_max_5)]
    subtask_desc_5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", "thinking of subtask 4", "answer of subtask 4"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking_5, answer_5 = await agent([taskInfo, thinking_4, answer_4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking_4, answer_4] + all_thinking_5[r-1] + all_answer_5[r-1]
                thinking_5, answer_5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, converting intermediate to final product and selecting correct choice, thinking: {thinking_5.content}; answer: {answer_5.content}")
            all_thinking_5[r].append(thinking_5)
            all_answer_5[r].append(answer_5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_5, answer_5 = await final_decision_agent_5([taskInfo] + all_thinking_5[-1] + all_answer_5[-1], "Sub-task 5: Make final decision on the correct product from given choices.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final product selection, thinking: {thinking_5.content}; answer: {answer_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_5.content}; answer - {answer_5.content}")
    subtask_desc_5['response'] = {"thinking": thinking_5, "answer": answer_5}
    logs.append(subtask_desc_5)
    print("Step 5: ", sub_tasks[-1])
    
    cot_instruction_6 = "Sub-task 6: Perform a consistency check comparing the predicted product’s stereochemistry and numbering against all given multiple-choice options to confirm alignment and identify the correct product, based on output from Sub-task 5."
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc_6 = {
        "subtask_id": "subtask_6",
        "instruction": cot_instruction_6,
        "context": ["user query", "thinking of subtask 5", "answer of subtask 5"],
        "agent_collaboration": "CoT"
    }
    thinking_6, answer_6 = await cot_agent_6([taskInfo, thinking_5, answer_5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, performing consistency check on predicted product, thinking: {thinking_6.content}; answer: {answer_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_6.content}; answer - {answer_6.content}")
    subtask_desc_6['response'] = {"thinking": thinking_6, "answer": answer_6}
    logs.append(subtask_desc_6)
    print("Step 6: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking_6, answer_6, sub_tasks, agents)
    return final_answer, logs
